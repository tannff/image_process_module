#导入OpenCV库
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def FLANN(img1, img2):


    #转化为灰度图，减少后续计算量
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # diff = cv2.absdiff(img1_gray, img2_gray)
    
    # overlap_mask = diff < 25
    # overlap_pixels = np.sum(overlap_mask)
    # total_pixels = img1_gray.shape[0] * img1_gray.shape[1]
    
    # overlap_rate = overlap_pixels / total_pixels
    # print("重叠率:", overlap_rate)

    #创建特征提取器对象
    sift = cv2.SIFT_create()
    #orb = cv2.ORB_create(nfeatures=1000)  # 限制特征点数量
    
    #检测关键点并计算描述子
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    #在图像上绘制关键点
    for kp in kp1:
        cv2.circle(img1_gray, (int(kp.pt[0]), int(kp.pt[1])), 1, (255, 0, 0), 1)
    for kp in kp2:
        cv2.circle(img2_gray, (int(kp.pt[0]), int(kp.pt[1])), 1, (255, 0, 0), 1)

    #使用FLANN角点匹配进行匹配
    #设置FLANN匹配器参数，定义FLANN匹配器，使用 KNN 算法实现匹配
    indexParams = dict(algorithm=0, trees=5)   #使用KD树算法
    searchParams = dict(checks=50)   #设置搜索的节点数目

    flann = cv2.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(des1,des2,k=2)  #使用KNN算法进行匹配

    #筛选匹配
    matchesMask = [[0,0] for i in range(len(matches))]  #初始化掩码
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:    #0.7和0.75是常用的经验值
            matchesMask[i] = [1,0]   #标记为好匹配
            
    #计算好匹配点数量和图像重叠率
    good_matches_count = sum(1 for mask in matchesMask if mask[0] == 1)

    overlap_rate = good_matches_count / min(len(kp1), len(kp2)) * 100

    print("重叠率: {:.2f}%".format(overlap_rate))
    print("关键点数：{}, 匹配点数：{}".format(len(kp1), good_matches_count))
    
    #绘制前1000个匹配点
    drawParams = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask[:1000],
                       flags = 0)
    image_sign = cv2.drawMatchesKnn(img1_gray,kp1,img2_gray,kp2,matches[:1000],None,**drawParams)

    #获取一一对应的匹配点坐标
    src_pts = []  #图像1中的点
    dst_pts = []  #图像2中的点
    for i, (m, n) in enumerate(matches):
        if matchesMask[i][0] == 1:  #如果是好匹配
            #图像1中的点
            src_pts.append(kp1[m.queryIdx].pt)
            #图像2中的点（对应的匹配点）
            dst_pts.append(kp2[m.trainIdx].pt)
    
    #转换为numpy数组
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    # print("=" * 50)
    # print("匹配点对应关系:")
    # print("图像1坐标 -> 图像2坐标")
    # for i in range(len(src_pts)):  #打印匹配点对应关系
    #     print(f"{src_pts[i][0]} -> {dst_pts[i][0]}")
    print(f"总计: {len(src_pts)} 对匹配点")

    #计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)   #使用RANSAC算法计算单应性矩阵

    return H, good_matches_count

def Switch_two_image(img1, img2):

    start_time = time.time()

    H, match_count = FLANN(img1, img2)
    
    #使用单应性矩阵对图像进行变换
    h1, w1 = img1.shape[:2]  
    h2, w2 = img2.shape[:2]   #获取图像高度和宽度

    corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners_img1, H)
    
    #计算新画布的大小
    all_corners = np.concatenate((corners_img1, transformed_corners), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    #计算平移矩阵
    tx, ty = -x_min, -y_min
    translation_matrix = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    
    #应用平移
    H_translated = translation_matrix.dot(H)

    #变换图像1
    warped_img1 = cv2.warpPerspective(img1, H_translated, (x_max-x_min, y_max-y_min))
    
    #创建结果图像
    result_width = x_max - x_min
    result_height = y_max - y_min
    result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
       
    #放置变换后的图像1
    h_warp, w_warp = warped_img1.shape[:2]
    result[0:h_warp, 0:w_warp] = warped_img1
    
    #放置图像2
    result[ty:ty+h2, tx:tx+w2] = img2
    end_time = time.time()
    print("Execution time:", (end_time - start_time)*1000, "milliseconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    return result

def Switch_multiple_image(images):

    num_images = len(images)
    print(f"总共需要拼接 {num_images} 张图像。")
    
    if num_images < 2:
        print("至少需要2张图像进行拼接")
        return None
    
    #从第一张图像开始拼接
    panorama = images[0]
    
    for i in range(1, num_images):
        print(f"正在拼接第 {i+1} 张图像...")
        panorama = Switch_two_image(panorama, images[i])
        
        print(f"第 {i+1} 张图像拼接完成")

        #显示当前拼接结果
        cv2.imshow("Final Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #保存结果
        cv2.imwrite("panorama_result.jpg", panorama)
        print("全景图已保存为 panorama_result.jpg")
    
    return panorama


def load_and_validate_images(paths):

    #加载并验证图像

    images = []
    valid_paths = []
    
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
            print(f"成功加载第{i+1}张图像: {path} - 尺寸: {img.shape}")
        else:
            print(f"加载失败: {path}")
    
    return images, valid_paths

if __name__ == '__main__':

    #加载n张待拼接图像
    paths = [r"E:\C300SM\lujing_guihua\11\20251119-145913.jpg", 
             r"E:\C300SM\lujing_guihua\11\20251119-145935.jpg",
             r"E:\C300SM\lujing_guihua\11\20251119-145939.jpg"]
    
    # 加载并验证图像
    images, valid_paths = load_and_validate_images(paths)
    
    if len(images) >= 2:
        final_panorama = Switch_multiple_image(images)
        
        if final_panorama is not None:
            print("图像拼接完成！")
        else:
            print("图像拼接失败！")
    else:
        print("有效的图像数量不足，无法进行拼接")
    
    # plt.imshow(resultImage)
    # plt.show()