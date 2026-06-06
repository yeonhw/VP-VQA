import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default=r'E:/NR-360VQA/RAW', type=str, help='Path to input videos') # '/dockerdata/CSIQ_YUV' ./dataset/CSIQ/csiq_all  J_raw10g_rotate_yuv
    parser.add_argument('--score_file_path', default='./dataset/VQAODV/odv_720_frtest_g6.json', type=str, help='Path to input subjective score')  # odv_720_frtest_g6.json  './dataset/CSIQ/csiq_subj_score.json' for_test.json
    parser.add_argument('--load_model', default='./save/model_videoset_v3.pt', type=str, help='Path to load checkpoint') # ./save/model_videoset_v3.pt model_csiq0.pt_20
    parser.add_argument('--save_model', default='./save/model_weight_4&15lap&fov30.pt', type=str, help='Path to save checkpoint') # model_csiq0.pt
    parser.add_argument('--log_file_name', default='./log/odv-weight_4&15lap&fov30_test2.log', type=str, help='Path to save log') # './log/run.log' csiq_test3.log

    parser.add_argument('--channel', default=1, type=int, help='channel number of input data, 1 for Y channel, 3 for YUV')
    parser.add_argument('--size_x', default=112, type=int, help='patch size x of segment')
    parser.add_argument('--size_y', default=112, type=int, help='patch size y of segment')
    parser.add_argument('--stride_x', default=80, type=int, help='patch stride x between segments')
    parser.add_argument('--stride_y', default=80, type=int, help='patch stride y between segments')
    parser.add_argument('--stride_u', default=45, type=int, help='patch stride u between segments') # 30
    parser.add_argument('--stride_v', default=30, type=int, help='patch stride v between segments') # 15

    parser.add_argument('--learning_rate', default=3e-4, type=float, help='learning rate') # default=3e-4
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='L2 regularization')  # default=1e-2
    parser.add_argument('--epochs', default=200, type=int, help='epochs to train')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_opts()
    print(args)
