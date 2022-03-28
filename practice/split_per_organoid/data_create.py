import os
import random
import copy
import cv2
import numpy as np
from tqdm import tqdm


class ImgCreate():
    def __init__(self, fg_image, bg_image, out_img_path, txt_path, bg_use_times=1, grid_x_num=4,
                 grid_y_num=4, min_fg_times=1, max_fg_times=3, write_label=False):
        assert bg_use_times > 0
        assert min_fg_times > 0
        assert max_fg_times <= grid_x_num * grid_y_num
        assert min_fg_times < max_fg_times
        self.write_label = write_label
        self.use_times = bg_use_times
        self.fg_image = fg_image
        self.bg_image = bg_image
        self.out_img_path = out_img_path
        self.txt_path = txt_path
        self.grid_x_num = grid_x_num
        self.grid_y_num = grid_y_num
        self.min_times = min_fg_times
        self.max_times = max_fg_times
        if os.path.exists(self.txt_path):
            os.remove(self.txt_path)
        self.Create()

    def WriteTxt(self, txt_dir, img_name, nums, label_list):
        file = open(txt_dir, 'a+')
        file.write(img_name + '\n')
        file.write(str(nums) + '\n')
        for content in label_list:
            for info in content:
                file.write(str(info) + ' ')
            file.write('\n')
        file.close()

    def MapPic(self, fg_image, bg_image, gridx_num, gridy_num, ban_coordinate):
        fg_height, fg_width, _ = fg_image.shape
        bg_height, bg_width, _ = bg_image.shape
        stride_x = int(bg_width / gridx_num)
        stride_y = int(bg_height / gridy_num)
        grid_x = int(stride_x * random.randint(0, gridx_num - 1))  # 背景图左上角x坐标
        grid_y = int(stride_y * random.randint(1, int(gridy_num / 1.8)))
        while (grid_x, grid_y) in ban_coordinate:
            grid_x = int(stride_x * random.randint(0, gridx_num - 1))
            grid_y = int(stride_y * random.randint(1, int(gridy_num / 1.8)))

        coord_x = grid_x + random.randint(0, max(stride_x - fg_width - 1, 0))  # 前景图所在位置的随机x坐标
        coord_y = grid_y + random.randint(0, max(stride_y - fg_height - 1, 0))  # 前景图所在位置的随机y坐标
        if coord_x + fg_image.shape[1] > bg_image.shape[1]:
            coord_x = bg_image.shape[1] - fg_image.shape[1]
        if coord_y + fg_image.shape[0] > bg_image.shape[0]:
            coord_y = bg_image.shape[0] - fg_image.shape[0]
        tmp_img = bg_image[coord_y:coord_y + fg_height, coord_x:coord_x + fg_width, :]
        fg_image[fg_image <= 30] = tmp_img[fg_image <= 30]
        bg_image[coord_y:coord_y + fg_height, coord_x:coord_x + fg_width, :] = fg_image  # 前景图贴到背景图上
        return bg_image, (grid_x, grid_y), (coord_x, coord_y, fg_width, fg_height)

    def GetImgsDic(self, image_dir):
        imgs_dic = {}
        for img_name in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir, img_name))
            imgs_dic[img_name] = img  # 图片的数组形式添加到列表中
        return imgs_dic

    def Create(self):
        fg_image = self.GetImgsDic(self.fg_image)
        fg_dic = fg_image
        bg_image = self.GetImgsDic(self.bg_image)
        bg_dic = bg_image
        for bg_img_key in bg_dic:
            for i in tqdm(range(self.use_times)):
                txt_name = bg_img_key[:-4] + "_" + str(i) + '.txt'
                bg_img = copy.deepcopy(bg_dic[bg_img_key])  # 复制背景图
                ban_coordinate = []
                times = random.randint(self.min_times, self.max_times)
                label_list = []
                for fg_img_key in random.sample(fg_dic.keys(), times):
                    label = 0
                    fg_img = fg_dic[fg_img_key]
                    bg_img, ban_coord, map_coord = self.MapPic(fg_img, bg_img, self.grid_x_num, self.grid_y_num,
                                                               ban_coordinate)
                    ban_coordinate.append(ban_coord)
                    if self.write_label:
                        content = [label, map_coord[0], map_coord[1], map_coord[2], map_coord[3]]
                    else:
                        content = map_coord
                    label_list.append(content)
                img_name = txt_name[:-4] + '.png'
                cv2.imwrite(os.path.join(out_img_path, img_name), bg_img)


if __name__ == '__main__':
    fg_image = r'D:\tmp\data_pix2pix\GAN_G\crop_elastic\big'
    bg_image = r'D:\tmp\black_img'
    out_img_path = r'D:\tmp\data_pix2pix\GAN_G\compose_results\big_size_elastic\test1'
    txt_path = r'D:\tmp\data_pix2pix\GAN_G\label.txt'
    create_data_num = 20000
    # times=random.randint(6,15)
    ImgCreate(fg_image, bg_image, out_img_path, txt_path, bg_use_times=create_data_num)
