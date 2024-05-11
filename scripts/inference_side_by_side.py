from argparse import Namespace
import os
import time
import cv2
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, VideoFileClip
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import warnings

sys.path.append(".")
sys.path.append("..")

warnings.filterwarnings("ignore", category=DeprecationWarning)


from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
	test_opts = TestOptions().parse()

	test_opts.exp_dir = './experiment_01'
	test_opts.checkpoint_path = './experiment_01/checkpoints/best_model.pt'
	test_opts.data_path = './test_data'
	test_opts.test_batch_size = 4
	test_opts.test_workers = 4
	test_opts.target_age = '0,10,20,30,40,50,60,70,80,90,100'

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_side_by_side')
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

	print(f'Loading dataset for {opts.dataset_type}')
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path, transform=transforms_dict['transform_inference'], opts=opts, return_path=True)
	dataloader = DataLoader(dataset, batch_size=opts.test_batch_size, shuffle=False, num_workers=int(opts.test_workers), drop_last=False)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_time = []
	global_i = 0
	for input_batch, image_paths in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		batch_results = {}
		for idx, age_transformer in enumerate(age_transformers):
			with torch.no_grad():
				input_age_batch = [age_transformer(img.cpu()).to('cuda') for img in input_batch]
				input_age_batch = torch.stack(input_age_batch)
				input_cuda = input_age_batch.cuda().float()
				tic = time.time()
				result_batch = run_on_batch(input_cuda, net, opts)
				toc = time.time()
				global_time.append(toc - tic)

				resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
				for i in range(len(input_batch)):
					result = tensor2im(result_batch[i])
					im_path = image_paths[i]
					input_im = log_image(input_batch[i], opts)
					if im_path not in batch_results.keys():
						batch_results[im_path] = np.array(input_im.resize(resize_amount))
					batch_results[im_path] = np.concatenate([batch_results[im_path], np.array(result.resize(resize_amount))], axis=1)

		for im_path, res in batch_results.items():
			image_name = os.path.basename(im_path)
			im_save_path = os.path.join(out_path_results, image_name)
			Image.fromarray(np.array(res)).save(im_save_path)
			global_i += 1


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


def clean_directory(dp):
	# 清空指定目录下所有内容，但是保留目录本身
	for filename in os.listdir(dp):
		file_path = os.path.join(dp, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)  # 删除文件或链接
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)  # 删除目录
		except Exception as e:
			print(f'Failed to delete {file_path}. Reason: {e}')

	print("Directory clean finish!")


def find_latest_image(dp):
	# 定义一个空列表，用于存储找到的图片文件
	latest_image = None
	latest_time = 0

	# 遍历目录下的所有文件和文件夹
	for entry in os.listdir(dp):
		# 构建完整的文件路径
		full_path = os.path.join(dp, entry)
		# 检查是否是文件且是图片格式
		if os.path.isfile(full_path) and entry.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
			# 获取文件的修改时间
			file_time = os.path.getmtime(full_path)
			# 比较时间，找到最新的文件
			if file_time > latest_time:
				latest_time = file_time
				latest_image = full_path

	return latest_image


def make_mp4(dp, fn, fe):
	'''
	用于合成演变视频以及后续的GIF动图
	:param dp: 合成视频的图片序列所在的目录
	:param fn: 图片纯文件名
	:param fe: 图片后缀名（扩展名）
	:return:
	'''
	folder_path = dp
	filename_template = fn + '_{0}' + fe
	# 验证图片路径是否获得成功
	print('合成视频的图片序列所在目录如下: ')
	print(folder_path)
	print('以图片名检测图片序列，图片名模板如下: ')
	print(filename_template)

	# 按照模板排序，0～100岁生成图像而不包含原图，所以共11张人脸图像
	image_files = [os.path.join(folder_path, filename_template.format(i)) for i in range(11)]
	print('合成视频的图片序列如下: ')
	print(image_files)

	# 正向剪辑序列
	forward_clip = ImageSequenceClip(image_files, fps=1, with_mask=False, load_images=True)
	# 反向剪辑序列
	reverse_clip = ImageSequenceClip(image_files[::-1], fps=1, with_mask=False, load_images=True)
	# 合成最终剪辑序列
	final_clip = concatenate_videoclips([forward_clip, reverse_clip])

	# 保存视频的目录
	video_filename = os.path.join('./result', fn + '.mp4')
	# 保存视频
	final_clip.write_videofile(video_filename, codec='libx264', audio=False)
	print("Video finish!")


def delete_mp4_files(dp):
	# 删除指定目录下所有 .mp4 文件
	# 遍历目录中的文件
	for filename in os.listdir(dp):
		if filename.endswith('.mp4'):  # 检查文件扩展名是否为.mp4
			file_path = os.path.join(dp, filename)  # 获取完整的文件路径
			os.remove(file_path)  # 删除文件
			print(f"Deleted: {file_path}")

	print("No MP4 file exist!")


def crop(dp, start_x, start_y, end_x, end_y):
	'''
	:param dp: 被裁剪图像的路径
	:param start_x: 人脸检测区域起始横坐标
	:param start_y: 人脸检测区域起始纵坐标
	:param end_x: 人脸检测区域终点横坐标
	:param end_y: 人脸检测区域终点纵坐标
	:return:
	'''
	# 获取被处理图片
	image_path = dp
	# 使用 np.fromfile 读取图片文件到数组中
	image_data = np.fromfile(image_path, dtype=np.uint8)
	# 使用 cv2.imdecode 解码图片数据到图片数组
	image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

	# 如果图片名含有中文，则下面这句代码报错
	# image = cv2.imread(image_path)

	file_name = os.path.basename(dp)
	f_n, f_e = os.path.splitext(file_name)

	image_height, image_width = image.shape[:2]
	print(f'image_height: {image_height}')
	print(f'image_width: {image_width}')

	start_x = max(0, min(start_x, image_width))
	start_y = max(0, min(start_y, image_height))
	end_x = max(start_x, min(end_x, image_width))
	end_y = max(start_y, min(end_y, image_height))

	# 以1024为模板，如果模板采用更大的数字则调整精度更佳，反之则更粗略
	template = 1024
	# 人脸区域下界（可调整）
	ruler_end = 945.5
	# 人脸区域上界（可调整）
	ruler_start = 148.5

	# 数学建模
	crop_start_y = int(((start_y / ruler_start) - (end_y / ruler_end)) / ((1 / ruler_start) - (1 / ruler_end)))
	print(f'crop_start_y: {crop_start_y}')
	crop_height = int((start_y - crop_start_y) * (template / ruler_start))
	print(f'crop_height: {crop_height}')
	crop_width = crop_height
	print(f'crop_width: {crop_width}')
	crop_end_y = crop_start_y + crop_height
	print(f'crop_end_y: {crop_end_y}')

	med_x = (start_x + end_x) / 2
	print(f'med_x: {med_x}')
	crop_start_x = int(med_x - (crop_width / 2))
	print(f'crop_start_x: {crop_start_x}')
	crop_end_x = int(med_x + (crop_width / 2))
	print(f'crop_end_x: {crop_end_x}')

	if crop_start_x >= 0 and crop_start_y >= 0 and crop_end_x <= image_width and crop_end_y <= image_height:
		# 检查并修正坐标以避免超出图像边界
		crop_start_x = max(0, min(crop_start_x, image_width))
		crop_start_y = max(0, min(crop_start_y, image_height))
		crop_end_x = max(crop_start_x, min(crop_end_x, image_width))
		crop_end_y = max(crop_start_y, min(crop_end_y, image_height))

		# 实际从原图上裁剪出的图像
		cropped_image = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
	else:
		# 计算裁剪区域的实际坐标，防止超出图像边界
		left = max(0, min(crop_start_x, image_width))
		top = max(0, min(crop_start_y, image_height))
		right = max(crop_start_x, min(crop_end_x, image_width))
		bottom = max(crop_start_y, min(crop_end_y, image_height))

		# 实际从原图上裁剪出的图像
		cropped_image = image[top:bottom, left:right]

		# 创建一个全白色背景 Numpy 数组，形状为 (crop_height, crop_width, 3)
		# crop_height 和 crop_width 是最终理应（理想）得到的裁剪图像大小
		# 其中3表示 RGB 三个颜色通道
		white_background_array = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
		# cv2.imshow("White background", white_background_array)

		# 计算图像的中心点
		cropped_image_center_x = cropped_image.shape[1] // 2
		cropped_image_center_y = cropped_image.shape[0] // 2
		print(f'cropped_image_center_x: {cropped_image_center_x}')
		print(f'cropped_image_center_y: {cropped_image_center_y}')

		# 计算白色北京的中心点
		# crop_height 和 crop_width 是最终理应（理想）得到的裁剪图像大小
		background_center_x = crop_width // 2
		background_center_y = crop_height // 2
		print(f'background_center_x: {background_center_x}')
		print(f'background_center_y: {background_center_y}')

		# 计算偏移量
		offset_x = background_center_x - cropped_image_center_x
		offset_y = background_center_y - cropped_image_center_y

		# 确保图像粘贴不会超过白色背景的边界
		offset_x = max(0, offset_x)
		offset_y = max(0, offset_y)
		print(f'offset_x: {offset_x}')
		print(f'offset_y: {offset_y}')

		# 将图片粘贴到白色背景上
		if (offset_x + cropped_image.shape[1]) <= crop_width and (offset_y + cropped_image.shape[0]) <= crop_height:
			white_background_array[offset_y:offset_y + cropped_image.shape[0], offset_x:offset_x + cropped_image.shape[1]] = cropped_image
			cropped_image = white_background_array
		else:
			print("The image is larger than the background and will be clipped.")

	# crop_start_x = max(0, min(crop_start_x, image_width))
	# crop_start_y = max(0, min(crop_start_y, image_height))
	# crop_end_x = max(crop_start_x, min(crop_end_x, image_width))
	# crop_end_y = max(crop_start_y, min(crop_end_y, image_height))

	# cropped_image = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

	# 切割出待变换的人脸区域并保存到目标目录
	# cv2.imwrite('../test_data/' + file_name, cropped_image)
	# 同名保存
	cv2.imencode(f_e, cropped_image)[1].tofile('./test_data/' + file_name)


def aging_recognize():
	print("程序正在处理图片，完成后此界面将自动关闭，结果或自动显示，请稍候片刻......")

	AGE_LIST = ['(0-6)', '(8-15)', '(18-24)', '(25-32)', '(33-42)', '(45-55)', '(57-68)', '(70-90)']

	# 人脸检测模型路径
	prototxtPathF = './detector_models/face_detector/face_deploy.prototxt'
	weightsPathF = './detector_models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
	# 加载人脸检测模型
	faceNet = cv2.dnn.readNet(prototxtPathF, weightsPathF)

	# 年龄检测模型路径
	prototxtPathA = './detector_models/age_detector/age_deploy.prototxt'
	weightsPathA = './detector_models/age_detector/age_net.caffemodel'
	# 加载年龄检测模型
	ageNet = cv2.dnn.readNet(prototxtPathA, weightsPathA)

	# 设置图像图像
	target_path = './origin_image'
	# 调用函数寻找最新图片
	latest_image_path = find_latest_image(target_path)

	if latest_image_path:
		print(f'The latest image is: {latest_image_path}')

		# 获取被处理图片
		image_path = latest_image_path
		# 使用 np.fromfile 读取图片文件到数组中
		image_data = np.fromfile(image_path, dtype=np.uint8)
		# 使用 cv2.imdecode 解码图片数据到图片数组
		image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

		# 如果图片名含有中文，则下面这句代码报错
		# image = cv2.imread(image_path)

		# 使用os.path.basename获取文件名
		file_name = os.path.basename(latest_image_path)
		print(file_name)
		# 这两个步骤很重要
		f_n, f_e = os.path.splitext(file_name)
		print(f_n)
		print(f_e)

		src = image.copy()
		image_height, image_width = image.shape[:2]
		print(f'image_height: {image_height}')
		print(f'image_width: {image_width}')

		# 构造blob
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
		# 送入网络计算
		faceNet.setInput(blob)
		detect = faceNet.forward()

		# 检测锚框置信度
		confidences = []
		for i in range(0, detect.shape[2]):
			confidence = detect[0, 0, i, 2]
			confidences.append(confidence)

		print('整图人脸检测置信度数值列表如下: ')
		print(confidences)
		# 最大置信度数值代表最接近人脸的图像区域
		max_confidence = max(confidences)
		print(f'max_confidence: {max_confidence}')

		for i in range(0, detect.shape[2]):
			confidence = detect[0, 0, i, 2]
			# 过滤掉小的置信度,计算坐标,提取面部roi,构造面部blob特征
			# 本文设置置信度大于0.9的采用，否则就太不像人脸了
			if confidence > 0.9:
				if confidence == max_confidence:
					# 算锚框
					box = detect[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
					(startX, startY, endX, endY) = box.astype("int")
					face = image[startY:endY, startX:endX]
					faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

					# 预测年龄
					ageNet.setInput(faceBlob)
					predictions = ageNet.forward()
					i = predictions[0].argmax()
					age = AGE_LIST[i]
					ageConfidence = predictions[0][i]

					# 显示打印
					text = "age{}:{:.2f}%".format(age, ageConfidence * 100)
					print('人脸年龄估计结果如下: ')
					print(text)

					# 绘制显示框
					y = startY - 10 if startY - 10 > 10 else startY + 10
					print(f'startX = {startX}')
					print(f'startY = {startY}')
					print(f'endX = {endX}')
					print(f'endY = {endY}')
					print(f'y = {y}')
					cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
					cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

					crop(image_path, startX, startY, endX, endY)
				else:
					continue

		# 保存人脸年龄估计结果到目标目录
		# cv2.imwrite('../aging_recognize/' + file_name, image)
		# 同名保存
		cv2.imencode(f_e, image)[1].tofile('./aging_recognize/' + file_name)
		# 清理origin_image目录下的图片内容，否则影响下一张图片的结果合成
		clean_directory(target_path)
		return f_n, f_e
	else:
		print('No images found.')
		return None


def aging_transform():
	print("程序正在处理图片，完成后此界面将自动关闭，结果或自动显示，请稍候片刻......")

	# 设置目标路径
	target_path = './test_data'

	# 调用函数寻找最新图片
	latest_image_path = find_latest_image(target_path)

	# 根据结果输出反馈
	if latest_image_path:
		print(f'The latest image is: {latest_image_path}')

		# 使用os.path.basename获取文件名
		file_name = os.path.basename(latest_image_path)
		print(file_name)
		# 这两个步骤很重要
		f_n, f_e = os.path.splitext(file_name)
		print(f_n)
		print(f_e)

		run()

		image_path = './experiment_01/inference_side_by_side/' + file_name
		image = Image.open(image_path)

		width, height = image.size
		print(f"width: {width}, height: {height}")
		segment_width = width // 12

		for i in range(1, 12):
			left = i * segment_width
			right = (i + 1) * segment_width
			segment = image.crop((left, 0, right, height))
			# 为了合成 GIF 动图的良好效果，在分割时并不把原图切入目录，原因在于后续合成视频和动图直接使用这个分割图像目录
			index = i - 1
			segment.save('./segment/' + f_n + f'_{index}' + f_e)

		print("Image segment finish!")

		# 图片文件夹路径
		# 以下文件为拼接图片的相关代码，目的是将人脸年龄变换单图拼接起来，但是不包含原图片，保存在 result 目录下
		# 由于人脸年龄变换结果已经产生，故下面这段代码可以选择保留或者注释掉，并不影响最终呈现
		segment_image_path = './segment'

		# 获取文件夹中所有图片文件的路径
		all_files = [os.path.join(segment_image_path, file) for file in os.listdir(segment_image_path)]
		# print(all_files)
		image_files = sorted(
			[f for f in all_files if f.endswith(f_e) and f.startswith('./segment/' + f_n + '_')],
			key=lambda x: int(x.split('_')[1].split('.')[0])
		)

		# 验证图片路径是否获得成功
		# print(len(image_files))
		# print(image_files)

		# 打开第一张图片获取宽度和高度
		first_image = Image.open(image_files[0])
		width, height = first_image.size

		# 创建一个新的空白图片，用于拼接后面的图片
		combined_image = Image.new('RGB', (width * 11, height))
		# 拼接图片
		for i, image_file in enumerate(image_files):
			image = Image.open(image_file)
			combined_image.paste(image, (i * width, 0))
		# 保存拼接后的图片
		combined_image.save('./result/' + file_name)

		print("Image blend finish!")

		# 生成变化视频
		make_mp4(segment_image_path, f_n, f_e)

		# 生成变化GIF动图
		new_clip = VideoFileClip('./result/' + f_n + '.mp4')
		new_clip_gif = new_clip.subclip(0, 22)
		new_clip_gif.write_gif('./result/' + f_n + '.gif')

		print("GIF finish!")

		# 清理test_data目录下的图片内容，否则影响下一张图片的结果合成
		clean_directory(target_path)
		# 清理segment目录下的图片内容，否则影响下一张图片的结果合成
		clean_directory(segment_image_path)
		# 结果路径
		result_path = './result'
		# 不保留 .mp4 文件
		delete_mp4_files(result_path)
	else:
		print('No images found.')


def show_result(fn, fe):
	# 定义文件路径
	aging_recognize_result = './aging_recognize/' + fn + fe
	aging_transform_result = './experiment_01/inference_side_by_side/' + fn + fe

	# 显示人脸年龄估计结果
	if os.path.exists(aging_recognize_result):
		aging_recognize_img = Image.open(aging_recognize_result)
		aging_recognize_img.show()

	# 显示人脸年龄变换结果
	if os.path.exists(aging_transform_result):
		aging_transform_img = Image.open(aging_transform_result)
		aging_transform_img.show()

	print("END!!!")


class ImageLoader(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('图片加载器')
		self.setGeometry(50, 50, 900, 900)
		self.setCentralWidget(self.create_main_widget())

	def create_main_widget(self):
		central_widget = QWidget()
		layout = QVBoxLayout()

		self.image_label = QLabel()
		self.image_label.setAlignment(Qt.AlignCenter)
		layout.addWidget(self.image_label)

		self.choose_button = QPushButton('开始')
		self.choose_button.clicked.connect(self.choose_image)
		layout.addWidget(self.choose_button)

		self.process_button = QPushButton('处理')
		self.process_button.clicked.connect(self.start_processing)
		layout.addWidget(self.process_button)

		central_widget.setLayout(layout)
		return central_widget

	def choose_image(self):
		image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
		if image_path:
			self.show_image(image_path)
			self.image_path = image_path

	def show_image(self, image_path):
		image = QImage(image_path)
		if image.isNull():
			self.image_label.setText("无法加载图片")
			return

		pixmap = QPixmap.fromImage(image)
		self.image_label.setPixmap(pixmap.scaled(900, 900, Qt.KeepAspectRatio))

		# 转存图片到目标文件夹
		target_folder = "./origin_image"
		target_path = target_folder + '/' + image_path.split('/')[-1]
		image.save(target_path)

	def start_processing(self):
		if hasattr(self, 'image_path'):	 # 检查是否选择了图片
			# 切换到下一个界面，这一步很重要
			self.processing_ui = ImageProcessing(self.image_path)
			self.processing_ui.show()
			self.processing_ui.start_processing()
			self.close()  # 关闭当前窗口
		else:
			self.image_label.setText('请先选择一张图片')


class CustomStdOut:
	def __init__(self, text_edit):
		self.text_edit = text_edit

	def write(self, message):
		self.text_edit.append(message)

	def flush(self):
		pass  # 我们不关心刷新操作


class ImageProcessing(QMainWindow):
	def __init__(self, image_path):
		super().__init__()
		self.setWindowTitle('图片处理中')
		self.setGeometry(250, 250, 800, 600)
		self.setCentralWidget(self.create_main_widget())

		# 创建一个处理线程
		self.thread = ImageProcessingThread(self.output_text_edit, self)
		# finished 是信号关联槽，线程执行完毕，界面自动跳转到下一个界面
		self.thread.finished.connect(self.handle_thread_finished)

	def create_main_widget(self):
		central_widget = QWidget()
		layout = QVBoxLayout()

		self.output_text_edit = QTextEdit()
		self.output_text_edit.setReadOnly(True)
		layout.addWidget(self.output_text_edit)

		# 重定向 stdout 到界面文本框
		sys.stdout = CustomStdOut(self.output_text_edit)

		central_widget.setLayout(layout)
		return central_widget

	def start_processing(self):
		# 启动线程
		self.thread.start()

	def handle_thread_finished(self):
		# 都是同一个 filename
		filename = self.thread.filename
		self.filename = filename
		self.face_age_reco_ui = FaceAgeRecognition(self.filename)
		self.face_age_reco_ui.show()
		# 关闭当前窗口
		self.close()


class ImageProcessingThread(QThread):
	# finished 是信号关联槽
	finished = pyqtSignal()

	def __init__(self, text_edit, main_window):
		super().__init__()
		self.text_edit = text_edit
		self.main_window = main_window
		self.filename = None

	def run(self):
		# 调用函数
		# 拿到文件名和文件扩展名，这样就可以轻松定位到文件
		file_n, file_e = aging_recognize()
		self.filename = file_n + file_e
		aging_transform()
		# show_result(file_n, file_e)
		# 通知线程结束
		self.finished.emit()


class FaceAgeRecognition(QMainWindow):
	def __init__(self, filename):
		super().__init__()
		# 这一步很重要很关键
		self.filename = filename
		self.setWindowTitle('人脸年龄估计')
		self.setGeometry(50, 50, 900, 900)
		self.setCentralWidget(self.create_main_widget('./aging_recognize/' + filename))

	def create_main_widget(self, image_path):
		central_widget = QWidget()
		layout = QVBoxLayout()

		pixmap = QPixmap(image_path).scaled(900, 900, Qt.KeepAspectRatio)  # 保持图片初始的纵横比
		if pixmap.isNull():
			print('无法加载图片')
			return

		self.label = QLabel()
		self.label.setPixmap(pixmap)
		layout.addWidget(self.label)

		self.complete_button = QPushButton('完成，下一个')
		self.complete_button.clicked.connect(self.start_transformation)
		layout.addWidget(self.complete_button)

		central_widget.setLayout(layout)
		return central_widget

	def start_transformation(self):
		# 创建并显示“人脸年龄变换”界面
		self.face_age_transformation = FaceAgeTransformation(self.filename)
		self.face_age_transformation.show()
		# 关闭当前窗口
		self.close()


class FaceAgeTransformation(QMainWindow):
	def __init__(self, filename):
		super().__init__()
		self.filename = filename
		self.setWindowTitle('人脸年龄变换')
		self.setGeometry(250, 250, 4800, 400)
		self.setCentralWidget(self.create_main_widget('./experiment_01/inference_side_by_side/' + filename))

	def create_main_widget(self, image_path):
		central_widget = QWidget()
		layout = QVBoxLayout()

		pixmap = QPixmap(image_path).scaled(4800, 4800, Qt.KeepAspectRatio)
		if pixmap.isNull():
			print('无法加载图片')
			return

		self.label = QLabel()
		self.label.setPixmap(pixmap)
		layout.addWidget(self.label)

		central_widget.setLayout(layout)
		return central_widget


def main():
	app = QApplication(sys.argv)
	loader_ui = ImageLoader()
	loader_ui.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
