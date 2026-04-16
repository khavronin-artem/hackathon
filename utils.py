import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# 1. Ваши параметры камеры (Интринсики и Дисторсия)
fx, fy = 648.56867562, 692.87140685
cx, cy = 359.56615446, 301.17102108
width, height = 720, 576

camera_matrix = np.array(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# distortion = [k1, k2, p1, p2, k3]
distortion = np.array(
    [-0.60048007, 0.58542388, -0.00536959, 0.00236961, -0.34971993],
    dtype=np.float32,
)

# Экстринсики (поворот от IMU к камере)
r_imu_cam = np.array(
    [[0.0, 0.12186934, 0.99254615], [1.0, 0.0, 0.0], [0.0, 0.99254615, -0.12186934]],
    dtype=np.float32,
)

# матрица перехода между базисом NED в openCV
m_ned_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

# Количество метров в одном пикселе изображения
GSD = 0.3917749350557746


def process_dataset(
    dataset_path: Path = Path("data/dataset"),
    output_path: Path = Path("dataset_bev"),
    max_distance: int = 2500,
    out_width: int = 1152,
    out_height: int = 1440,
) -> None:
    """Превращает фото с дрона в BEV.

        Замечание по поводу всего:

        Важное замечание по поводу координат. Есть координаты NED - x направлен вперед,
        z вниз - используется для дрона, openCV - z направлено в картинку, x направо.
        По этой причине матрица перехода от imu к cam `R_imu_cam` такая странная и она
        исправляется ниже.

        Параметр R в функции `cv2.initUndistortRectifyMap` должен принимать матрицу
        перехода от идеального базиса к камере, которая сейчас у нас, т.е.
        $P_{ideal} = R P_{cam}$

        У `R_imu_cam`, новая - это камера, а старая это IMU (от IMU к камере). Но в
        openCV, мы хотим смотреть как дрон, т.е. идеальная это IMU, а камера это камера,
        поэтому надо будет транспонировать.

        Ниже написана матрица M, которая переводит из системы координат NED в систему
        openCV.

        Размеры карты: 18-ый тайл, расрешение тайла 256x256, 49 - северная широта. GSD -
        это количество метров в одном пикселе


    Args:
            dataset_path (Path, optional): Путь к целевому датасету.
            Defaults to Path("data/dataset").
            output_path (Path, optional): Путь к преобразованным картинкам.
            Defaults to Path("dataset_bev").
            max_distance (int, optional): Максимальная дистанция до самых дальних объектов
            на картинке. Defaults to 2500.
            out_width (int, optional): Ширина выхода. Defaults to 1152.
            out_height (int, optional): Высота выхода. Defaults to 1440.

    Raises:
            ValueError: На картинке только небо
            ValueError: Нет нужного изображения
    """
    # map_resolution — это сколько метров в одном пикселе твоей Slippy Map
    # (например, 0.15)
    output_path.mkdir(parents=True, exist_ok=True)
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

    # 1. Устранение дисторсии (Undistortion)
    # Получаем оптимальную матрицу камеры (чтобы не потерять пиксели по краям после
    # обрезки)
    # roi = [x,y,w,h] - прямоугольник
    # alpha = 0.0 включает обрезку полученного изображения
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        imageSize=(width, height),
        alpha=0.0,
        newImgSize=(width, height),
    )
    print(new_camera_matrix)

    # zoom_factor = 1
    # new_camera_matrix[0, 0] *= zoom_factor
    # new_camera_matrix[1, 1] *= zoom_factor

    # current_K = new_camera_matrix.copy()

    # x, y, w, h = (0, 0, width, height)
    # if trim:
    #     x, y, w, h = roi
    #     current_K[0, 2] -= x
    #     current_K[1, 2] -= y

    print(folders[0:5])
    # для откладки делаем только для первых пяти
    for folder in folders[0:5]:
        out_folder = output_path / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        # Загружаем JSON
        json_file = next(folder.glob("*.json"))
        with json_file.open("r") as f:
            telemetry = cast("list[dict[str, float | int]]", json.load(f))

        images = sorted(folder.glob("*.jpeg"))

        for idx, img_path in enumerate(images):
            if idx >= len(telemetry):
                break

            frame_data = telemetry[idx]

            # 2. Вычисление матриц поворота с помощью SciPy
            # Углы IMU дрона. В авиации обычно порядок Yaw, Pitch, Roll (Z, Y, X)
            # (рыскание, тангаж, крен). Даны в радианах, не в градусах.
            # scipy ожидает порядок осей вращения. Предположим стандартную авиационную
            # (intrinsic) систему:
            r_st_imu = Rotation.from_euler(
                "zyx",
                [
                    0,  # ВРЕМЕННО УБРАЛ YAW!!!!
                    frame_data["pitch"],
                    frame_data["roll"],
                ],
                degrees=False,
            ).as_matrix()
            r_bev_st = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
            r_final = r_bev_st @ m_ned_cam.T @ r_st_imu @ r_imu_cam

            # крайние пиксели изображения
            # corners = np.array(
            #     [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
            #     dtype=np.float32,
            # )
            x_roi, y_roi, w_roi, h_roi = roi
            # Углы ROI в пикселях исходного (неискажённого) изображения
            edge_points: list[list[int]] = []
            steps = 50
            # Верхняя и нижняя границы
            for x_p in np.linspace(x_roi, x_roi + w_roi - 1, steps, dtype=np.int32):
                edge_points.append([x_p, y_roi])
                edge_points.append([x_p, y_roi + h_roi - 1])
            # Левая и правая границы
            for y_p in np.linspace(y_roi, y_roi + h_roi - 1, steps, dtype=np.int32):
                edge_points.append([x_roi, y_p])
                edge_points.append([x_roi + w_roi - 1, y_p])
            # corners = np.array(
            #     [
            #         [x_roi, y_roi],
            #         [x_roi + w_roi - 1, y_roi],
            #         [x_roi, y_roi + h_roi - 1],
            #         [x_roi + w_roi - 1, y_roi + h_roi - 1],
            #     ],
            #     dtype=np.float32,
            # )
            corners = np.array(edge_points, dtype=np.float32)
            undistorted = cv2.undistortPoints(
                corners.reshape(-1, 1, 2),
                camera_matrix,
                distortion,
                P=np.eye(3),
            )
            # 4 луча по строчкам, их однороные координаты по столбцам, shape=(4,3)
            rays_cam = np.hstack(
                [undistorted.reshape(-1, 2), np.ones((corners.shape[0], 1))]
            )
            # они же в координатах bev
            rays_bev = (r_final @ rays_cam.T).T

            valid_rays = rays_bev[:, 2] > 0
            if not np.any(valid_rays):
                raise ValueError(f"Only sky on the image - {str(img_path)}")

            # лучи это (t * x, t * y, t * z) (параметрическое задание), земля это
            # z = alt, тогда t = alt / z, координаты точки на земле (t * x, t * y)
            t = frame_data["alt"] / rays_bev[valid_rays, 2]  # dz > 0
            # (4,3), координаты (X, Y, alt)
            points_ground = t[:, np.newaxis] * rays_bev[valid_rays, :]
            points_2d_cord = points_ground[:, 0:-1]

            # исправляем далекие точки
            distances = np.sqrt((points_2d_cord**2).sum(axis=1))
            far_ind = distances > max_distance
            if np.any(far_ind):
                points_2d_cord[far_ind] = (
                    points_2d_cord[far_ind] / distances[far_ind, np.newaxis]
                ) * max_distance
            # additional_points = (points_2d_cord / distances) * max_distance
            # points_2d_cord: NDArray[float64] = np.vstack([points_2d_cord, additional_points])

            x_min, x_max = np.min(points_2d_cord[:, 0]), np.max(points_2d_cord[:, 0])
            y_min, y_max = np.min(points_2d_cord[:, 1]), np.max(points_2d_cord[:, 1])
            width_m = x_max - x_min
            height_m = y_max - y_min

            # считаем метры на пиксель
            gsd_x = width_m / out_width
            gsd_y = height_m / out_height
            gsd = max(gsd_x, gsd_y)

            f_virtual = frame_data["alt"] / gsd

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            u_center = center_x / gsd
            v_center = center_y / gsd

            cx_bev = out_width / 2 - u_center
            cy_bev = out_height / 2 - v_center

            k_bev = np.array(
                [[f_virtual, 0, cx_bev], [0, f_virtual, cy_bev], [0, 0, 1]],
                dtype=np.float32,
            )

            mapx, mapy = cv2.initUndistortRectifyMap(
                cameraMatrix=camera_matrix,
                distCoeffs=distortion,
                R=r_final,
                newCameraMatrix=k_bev,
                size=(out_width, out_height),
                m1type=cv2.CV_32FC1,
            )

            # BGR - каналы, формат HWC
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Файл не найден: {str(img_path)}")

            bev_img = cv2.remap(
                img,
                mapx,
                mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # ВАЖНАЯ ЧАСТЬ, нужна образка или нет (см if trim выше)
            # undistorted_img = undistorted_img[y : y + h, x : x + w]

            _ = cv2.imwrite(
                str(out_folder / f"{img_path.stem}_bev{img_path.suffix}"),
                bev_img,
            )


if __name__ == "__main__":
    process_dataset()
