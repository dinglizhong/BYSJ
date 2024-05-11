from scripts.inference_side_by_side import main

# 必须注意，当我们在一个py文件调用另一个py文件的函数时，程序中的相对路径以目前正所出的py文件为准
# 所以要对被调用的py文件中的相对路径做相应的修改，这点很重要
if __name__ == '__main__':
    main()
