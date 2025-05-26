import numpy as np
import math
from math import cos, sin

from scipy import optimize



####################### Extrinsic Projection Model #######################

def projection_move_by_lm(keypoints, lm_512, lm_index):
    keypoints, _ = bbox_center_normalize(dsa_points=keypoints, imager_size=(512, 512))
    keypoints = lm_512 - keypoints[lm_index] + keypoints
    return keypoints


def bbox_center_normalize(dsa_points, imager_size):
    min_x, min_y = dsa_points.min(axis=0)
    max_x, max_y = dsa_points.max(axis=0)

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)

    bbox_center = np.array([center_x, center_y])
    imager_center = np.array(imager_size) / 2

    delta_coords = imager_center - bbox_center
    return dsa_points + delta_coords, delta_coords


def bbox_center_normalize_cta_dsa(dsa_points, cta_2d_points, imager_size):
    min_x, min_y = dsa_points.min(axis=0)
    max_x, max_y = dsa_points.max(axis=0)

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)

    bbox_center = np.array([center_x, center_y])
    imager_center = np.array(imager_size) / 2

    delta_coords = imager_center - bbox_center
    # TODO：注意+=会改变原本的值
    dsa_points += delta_coords
    cta_2d_points += delta_coords
    return dsa_points, cta_2d_points


def axis_angle_to_quaternion(axis, theta):
    assert axis.ndim == 1 and axis.size == 3

    # quaternion is in form of [x, y, z, w]
    quaternion = np.zeros(4, dtype=np.float32)
    quaternion[3] = 1

    # normalize rotation axis
    norm = np.linalg.norm(axis)
    if norm == 0:
        print("Warning: axis is zero so return identity quaternion")
        return quaternion
    axis = axis / norm

    cos_theta_2 = cos(theta / 2)
    sin_theta_2 = sin(theta / 2)
    w = cos_theta_2
    x = axis[0] * sin_theta_2
    y = axis[1] * sin_theta_2
    z = axis[2] * sin_theta_2

    quaternion[0] = x
    quaternion[1] = y
    quaternion[2] = z
    quaternion[3] = w

    return quaternion


def quaternion_to_rotation_matrix(quaternion):
    assert quaternion.ndim == 1 and quaternion.size == 4

    rotation_matrix = np.identity(3, dtype=np.float32)

    # normalize quaternion
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        print("Warning: quaternion is zero so return identity rotation matrix")
        return rotation_matrix
    quaternion = quaternion / norm

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    rotation_matrix[0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[0, 1] = 2 * (x * y - z * w)
    rotation_matrix[0, 2] = 2 * (x * z + y * w)

    rotation_matrix[1, 0] = 2 * (x * y + z * w)
    rotation_matrix[1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[1, 2] = 2 * (y * z - x * w)

    rotation_matrix[2, 0] = 2 * (x * z - y * w)
    rotation_matrix[2, 1] = 2 * (y * z + x * w)
    rotation_matrix[2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix


def axis_angle_to_rotation_matrix(axis, theta):
    quaternion = axis_angle_to_quaternion(axis, theta)

    return quaternion_to_rotation_matrix(quaternion)


def extrinsic_euler_angles_to_rotation_matrix(angles_in_degrees: np.array):
    [alpha, beta, gamma] = [math.radians(angle) for angle in angles_in_degrees]

    x_axis = np.array([1, 0, 0], dtype=np.float32)
    rotation_x = axis_angle_to_rotation_matrix(x_axis, alpha)

    y_axis = np.array([0, 1, 0], dtype=np.float32)
    rotation_y = axis_angle_to_rotation_matrix(y_axis, beta)

    z_axis = np.array([0, 0, 1], dtype=np.float32)
    rotation_z = axis_angle_to_rotation_matrix(z_axis, gamma)

    rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation


def extrinsic_euler_angles_to_rotation_matrix_direct(angles_in_degrees: np.array):
    [alpha, beta, gamma] = [math.radians(angle) for angle in angles_in_degrees]

    rotation_x = np.array([[1, 0, 0],
                           [0, cos(alpha), -sin(alpha)],
                           [0, sin(alpha), cos(alpha)]], dtype=np.float32)

    rotation_y = np.array([[cos(beta), 0, sin(beta)],
                            [0, 1, 0],
                            [-sin(beta), 0, cos(beta)]], dtype=np.float32)

    rotation_z = np.array([[cos(gamma), -sin(gamma), 0],
                            [sin(gamma), cos(gamma), 0],
                            [0, 0, 1]], dtype=np.float32)

    rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation


def cta_2_dsa_rotation_matrix_wbex(angles_in_degrees: np.array, direct=True):
    x_axis = np.array([0, 0, 1], dtype=np.float32)
    y_axis = np.array([-1, 0, 0], dtype=np.float32)
    z_axis = np.array([0, -1, 0], dtype=np.float32)

    rotation_1 = np.concatenate(([x_axis], [y_axis], [z_axis]))
    rotation_1 = np.transpose(rotation_1)
    if direct:
        rotation_2 = extrinsic_euler_angles_to_rotation_matrix_direct(
            angles_in_degrees)
    else:
        rotation_2 = extrinsic_euler_angles_to_rotation_matrix(
            angles_in_degrees)

    # rotation_matrix[3x3] = [x_axis[3x1]; y_axis[3x1]; z_axis[3x1]]
    # here x_axis[3x1], y_axis[3x1], z_axis[3x1] are column vectors
    rotation_matrix = np.matmul(rotation_1, rotation_2)
    return rotation_matrix


def cta_2_dsa_3d_wbex(heart_center: np.array,
                      angles_in_degrees: np.array,
                      SOD: np.float32,
                      points_cta: np.array):
    # points_cta in the shape of [[x1, y1, z1], [x2, y2, z2], ...., [xn, yn, zn]]

    # 1. apply translation
    # heart_center in the shape of [1x3]
    # move the origin to the heart_center
    points_dsa = points_cta - np.array(heart_center)[None, :]

    # 2. apply rotation
    # get rotation matrix
    rotation_matrix = cta_2_dsa_rotation_matrix_wbex(angles_in_degrees, direct=True)


    # the rotation matrix takes form of the [x_axis[3x1]; y_axis[3x1]; z_axis[3x1]]
    # where the x_axis, y_axis, z_axis are three axes of the DSA C-ARM represented in
    # the CTA coordinate system, so we need to transpose the rotation matrix first
    # to convert the CTA coordinates to the DSA coordinates
    points_dsa = np.matmul(np.transpose(rotation_matrix), np.transpose(points_dsa))
    points_dsa = np.transpose(points_dsa)

    # 3. apply translation
    # move the origin to the X-ray source
    points_dsa = points_dsa + np.array([0, 0, SOD])[None, :]
    return points_dsa


def dsa_3d_to_2d_wbex(pixel_spacing: np.array,
                      principal_point: np.array,
                      SID: np.float32,
                      points_dsa_3d: np.array):
    points_dsa_x = points_dsa_3d[:, 0]
    points_dsa_y = points_dsa_3d[:, 1]
    points_dsa_z = points_dsa_3d[:, 2]

    cx = principal_point[0]
    cy = principal_point[1]

    pixel_spacing_x = pixel_spacing[0]
    pixel_spacing_y = pixel_spacing[1]

    dsa_2d_x = np.divide(points_dsa_x, points_dsa_z) * SID / pixel_spacing_x + cx
    dsa_2d_y = np.divide(points_dsa_y, points_dsa_z) * SID / pixel_spacing_y + cy

    points_dsa_2d = np.concatenate(([dsa_2d_x], [dsa_2d_y]))
    return np.transpose(points_dsa_2d)


def flip_dsa_points_wbex(dsa_points: np.array,
                         dsa_size: np.array):
    dsa_points_new = np.zeros(dsa_points.shape)
    dsa_points_new[:, 0] = dsa_size[0] - dsa_points[:, 1]
    dsa_points_new[:, 1] = dsa_size[1] - dsa_points[:, 0]
    return dsa_points_new


def cta_2_dsa_2d_wbex(cta_points: np.array,
                      param_dict: dict,
                      radius=None):
    heart_center = param_dict["Translation"]
    initial_rotation = param_dict["Rotation"]
    SOD = param_dict["SOD"]
    SID = param_dict["SID"]
    image_spacing = param_dict["DSAImagerSpacing"]
    image_size = param_dict["DSAImagerSize"]

    # Transfer cta 3d points to dsa coordinate
    cta_points_in_dsa_coordinates = cta_2_dsa_3d_wbex(np.asarray(heart_center), \
                                                      np.asarray(initial_rotation), \
                                                      SOD, \
                                                      cta_points)

    # Project cta points on 2d
    projected_2d_point = dsa_3d_to_2d_wbex(np.asarray(image_spacing), \
                                           np.asarray([int(image_size[0] / 2), int(image_size[1] / 2)]), \
                                           SID, \
                                           cta_points_in_dsa_coordinates)
    # Flip dsa points in projection plane
    dsa_points = flip_dsa_points_wbex(projected_2d_point, image_size)
    if radius is not None:
        radius_project = np.divide(SID, cta_points_in_dsa_coordinates[:, 2]) * radius / image_spacing[0]
        return dsa_points, radius_project
    return dsa_points


def ctamask_2_dsa_2d_wbex(cta_points: np.array,
                          param_dict: dict,
                          radius: np.array):
    heart_center = param_dict["Translation"]
    initial_rotation = param_dict["Rotation"]
    SOD = param_dict["SOD"]
    SID = param_dict["SID"]
    image_spacing = param_dict["DSAImagerSpacing"]
    image_size = param_dict["DSAImagerSize"]

    # Transfer cta 3d points to dsa coordinate
    cta_points_in_dsa_coordinates = cta_2_dsa_3d_wbex(np.asarray(heart_center), \
                                                      np.asarray(initial_rotation), \
                                                      SOD, \
                                                      cta_points)

    # Project cta points on 2d
    projected_2d_point = dsa_3d_to_2d_wbex(np.asarray(image_spacing), \
                                           np.asarray([int(image_size[0] / 2), int(image_size[1] / 2)]), \
                                           SID, \
                                           cta_points_in_dsa_coordinates)

    # Flip dsa points in projection plane
    dsa_points = flip_dsa_points_wbex(projected_2d_point, image_size)
    if radius is not None:
        radius_project = np.divide(SID, cta_points_in_dsa_coordinates[:, 2]) * radius / image_spacing[0]
        return dsa_points, radius_project, cta_points_in_dsa_coordinates, projected_2d_point
    return dsa_points



####################### Optimize Rigid Params by Matching Points #######################


def projection_using_optimized_param(cta_points, optimize_param):
    param_dict = {}
    param_dict["Rotation"] = optimize_param[0:3]
    param_dict['Translation'] = optimize_param[3:6]
    param_dict['SOD'] = optimize_param[6]
    param_dict['SID'] = optimize_param[7]
    param_dict["DSAImagerSpacing"] = optimize_param[10:]
    param_dict["DSAImagerSize"] = optimize_param[8:10]
    dsa_points = cta_2_dsa_2d_wbex(cta_points, param_dict)
    return dsa_points


def CTA_to_DSA_minimize_extrinsic(optimize_params, CTA_points, DSA_points, params_optimize, other_params):
    params = other_params['params']
    key = ["primary_angle", "secondary_angle", "roll_angle", "hx", "hy", "hz", "SOD", "SID"]
    n = 0
    for i in range(8):
        if key[i] in params_optimize.keys():
            params[i] = optimize_params[n]
            n = n + 1
    CTA_points_2d = projection_using_optimized_param(CTA_points, params)
    out = np.array(CTA_points_2d) - DSA_points
    out_squeeze = np.reshape(out, [out.size])
    return out_squeeze


def optimize_params_extrinsic(CTA_points, DSA_points, param_dict, params_optimize):
    """
    :param CTA_points: n*3
    :param DSA_points: n*2
    :param params: [primary,secondary_angle,roll_angle,hx,hy,hz,SOD,SID,size_x,size_y,spacing_x,spacing_y]
    :param params_optimize: {optimized_key:[min,max]}
    :param log_txt: log save_path
    :return:
    optimized_params:[]
    new_params:[primary,secondary_angle,roll_angle,hx,hy,hz,SOD,SID,size_x,size_y,spacing_x,spacing_y]
    CTA_points_2D:n*2
    raw_CTA_points_2D:n*2
    """
    key = ["primary_angle", "secondary_angle", "roll_angle", "hx", "hy", "hz", "SOD", "SID"]
    raw_CTA_points_2D = cta_2_dsa_2d_wbex(CTA_points, param_dict)
    heart_center = param_dict["Translation"]
    initial_rotation = param_dict["Rotation"]
    SOD = param_dict["SOD"]
    SID = param_dict["SID"]
    image_spacing = param_dict["DSAImagerSpacing"]
    image_size = param_dict["DSAImagerSize"]
    params = initial_rotation + heart_center + [SOD] + [SID] + image_size + image_spacing
    optimize_params = []
    lower_bound = []
    upper_bound = []
    for i in range(8):
        if key[i] in params_optimize.keys():
            optimize_params.append(params[i])
            lower_bound.append(params_optimize[key[i]][0])
            upper_bound.append(params_optimize[key[i]][1])
    other_params = {'params': params.copy()}
    optimized_result = optimize.least_squares(CTA_to_DSA_minimize_extrinsic,
                                              optimize_params, bounds=(
            lower_bound, upper_bound), jac="3-point", method="dogbox", \
                                              args=(CTA_points, DSA_points, params_optimize, other_params),
                                              loss="soft_l1")

    optimized_params = optimized_result['x']
    new_params = params.copy()
    n = 0
    for i in range(8):
        if key[i] in params_optimize.keys():
            new_params[i] = optimized_params[n]
            n = n + 1

    new_param_dict = {}
    new_param_dict["Rotation"] = new_params[0:3]
    new_param_dict['Translation'] = new_params[3:6]
    new_param_dict['SOD'] = new_params[6]
    new_param_dict['SID'] = new_params[7]
    new_param_dict["DSAImagerSpacing"] = param_dict["DSAImagerSpacing"]
    new_param_dict["DSAImagerSize"] = param_dict["DSAImagerSize"]

    return optimized_params, new_params


def optimization_by_matching_points(cta_3d_points, dsa_2d_points, initial_params):
    initial_rotation = initial_params["Rotation"]
    heart_center = initial_params['Translation']
    SOD = initial_params["SOD"]
    SID = initial_params["SID"]
    image_spacing = initial_params["DSAImagerSpacing"]
    image_size = initial_params["DSAImagerSize"]
    params = initial_rotation + heart_center + [SOD] + [SID] + image_size + image_spacing

    params_optimize = {
        "primary_angle": [params[0] - 40, params[0] + 40], \
        "secondary_angle": [params[1] - 40, params[1] + 40], \
        "roll_angle": [params[2] - 40, params[2] + 40], \
        "hx": [params[3] - 400, params[3] + 400], \
        "hy": [params[4] - 1000, params[4] + 1000], \
        "hz": [params[5] - 400, params[5] + 400]
    }
    optimized_params, params_new = \
        optimize_params_extrinsic(cta_3d_points, dsa_2d_points, initial_params, params_optimize)

    optimized_result = {}
    optimized_result["Rotation"] = params_new[0:3]
    optimized_result["Translation"] = params_new[3:6]
    optimized_result['SOD'] = initial_params['SOD']
    optimized_result['SID'] = initial_params['SID']
    optimized_result['DSAImagerSize'] = initial_params['DSAImagerSize']
    optimized_result['DSAImagerSpacing'] = initial_params['DSAImagerSpacing']

    return optimized_result


####################### Metrics Evaluating Optimization #######################

def get_norm(input_array):
    return np.linalg.norm(input_array, ord=None, axis=None, keepdims=False)


def metrics_l2_norm_angle_difference(params0, params1):
    l2_norm_angle_difference = get_norm(np.asarray([params1['Rotation'][0] - params0['Rotation'][0], \
                                                    params1['Rotation'][1] - params0['Rotation'][1], \
                                                    params1['Rotation'][2] - params0['Rotation'][2]]))

    return l2_norm_angle_difference


def metrics_mse(params0, params1, cta_points_3d, lm_delta=None):
    dsa_points_2d0 = cta_2_dsa_2d_wbex(cta_points_3d, params0)
    dsa_points_2d1 = cta_2_dsa_2d_wbex(cta_points_3d, params1)
    if lm_delta is not None:
        dsa_points_2d1 +=lm_delta
    mse = np.sqrt(np.square(np.subtract(dsa_points_2d0 / params0['DSAImagerSize'][0], \
                                        dsa_points_2d1 / params1['DSAImagerSize'][0])).mean()) * \
          params1['DSAImagerSize'][0] * 1.414

    return mse