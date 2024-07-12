class paths:
    smpl_file = 'models/SMPL_male.pkl'
    smpl_file_f = 'models/SMPL_female.pkl'

    aist_raw_dir = 'D:/_dataset/AIST/'  # e.g., keypoints2d, motions, cameras, splits, video
    aist_dir = 'data/dataset_work/AIST/'
    aist_tip_dir = 'data/dataset_work/AIST/tip'

    amass_raw_dir = 'D:/_dataset/AMASS/'  # e.g., ACCAD/ACCAD/*/*.npz
    amass_dir = 'data/dataset_work/AMASS/'

    totalcapture_raw_dir = 'D:/_dataset/TotalCapture/'
    totalcapture_dir = 'data/dataset_work/TotalCapture/'

    pw3d_raw_dir = 'D:/_dataset/3DPW/'
    pw3d_dir = 'data/dataset_work/3DPW/'
    pw3d_tip_dir = 'data/dataset_work/3DPW/tip'
    pw3d_pip_dir = 'data/dataset_work/3DPW/pip'

    offline_dir = 'data/dataset_work/live/'
    live_dir = 'data/sig_asia/live'

    weight_dir = 'data/weights/'

    occ_dir = 'F:\ShaohuaPan\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
    j_regressor_dir = 'data/dataset_work/J_regressor_h36m.npy'


class amass_data:
    train = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 'DanceDB', 'DFaust67', 'EKUT',
             'Eyes_Japan_Dataset', 'GRAB', 'HUMAN4D', 'KIT', 'MPI_Limits', 'TCD_handMocap', 'TotalCapture']
    val = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'SOMA', 'WEIZMANN', 'Transitions_mocap', 'SSM_synced']
    test = []


# vctoolkit.skeletons
class HUMBIBody33:
    n_keypoints = 33

    labels = [
        'pelvis',  # 0
        'left_hip', 'right_hip',  # 2
        'lowerback',  # 3
        'left_knee', 'right_knee',  # 5
        'upperback',  # 6
        'left_ankle', 'right_ankle',  # 8
        'thorax',  # 9
        'left_toes', 'right_toes',  # 11
        'lowerneck',  # 12
        'left_clavicle', 'right_clavicle',  # 14
        'upperneck',  # 15
        'left_shoulder', 'right_shoulder',  # 17
        'left_elbow', 'right_elbow',  # 19
        'left_wrist', 'right_wrist',  # 21
        # the fake hand joints in SMPL are removed
        # following are extended keypoints
        'head_top', 'left_eye', 'right_eye',  # 24
        'left_hand_I0', 'left_hand_L0',  # 26
        'right_hand_I0', 'right_hand_L0',  # 28
        'left_foot_T0', 'left_foot_L0',  # 30
        'right_foot_T0', 'right_foot_L0',  # 32
    ]

    parents = [
        None,
        0, 0,
        0,
        1, 2,
        3,
        4, 5,
        6,
        7, 8,
        9,
        9, 9,
        12,
        13, 14,
        16, 17,
        18, 19,
        # extended
        15, 15, 15,
        20, 20,
        21, 21,
        7, 7,
        8, 8
    ]

    # the vertex indices of the extended keypoints
    extended_keypoints = {
        22: 411, 23: 2800, 24: 6260,
        25: 2135, 26: 2062,
        27: 5595, 28: 5525,
        29: 3292, 30: 3318,
        31: 6691, 32: 6718
    }


vel_scale = 3
tran_offset = [0, 0.25, 5]
mp_mask = [332, 2809, 2800, 455, 6260, 3634, 3621, 583, 4071, 45, 3557, 1873, 4123, 1652, 5177, 2235, 5670, 2673, 6133, 2319, 5782, 2746, 6191, 3138, 6528, 1176, 4662, 3381, 6727, 3387, 6787, 3226, 6624]
vi_mask = [1961, 5424, 1176, 4662, 411, 3021]
ji_mask = [18, 19, 4, 5, 15, 0]

class Live:
    camera_intrinsic = [[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]]
    camera_height = 480
    camera_width = 640
    camera_id = 0
    imu_addrs = [
    'D4:22:CD:00:36:03',
    'D4:22:CD:00:44:6E',
    'D4:22:CD:00:45:E6',
    'D4:22:CD:00:45:EC',
    'D4:22:CD:00:46:0F',
    'D4:22:CD:00:32:32',
    ]



class Pw3d_data:
    pw3d_occluded_sequences = [
        'courtyard_backpack',
        'courtyard_basketball',
        'courtyard_bodyScannerMotions',
        'courtyard_box',
        'courtyard_golf',
        'courtyard_jacket',
        'courtyard_laceShoe',
        'downtown_stairs',
        'flat_guitar',
        'flat_packBags',
        'outdoors_climbing',
        'outdoors_crosscountry',
        'outdoors_fencing',
        'outdoors_freestyle',
        'outdoors_golf',
        'outdoors_parcours',
        'outdoors_slalom',
    ]