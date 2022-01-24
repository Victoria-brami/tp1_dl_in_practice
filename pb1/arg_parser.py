import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', action='store_true')
    parser.add_argument('--data_dir', default='../../TP1/USPS/')
    parser.add_argument('--conv_layers_archi', action='store_true')
    parser.add_argument('--linear_layers_archi', action='store_true')
    parser.add_argument('--activations', action='store_true')
    parser.add_argument('--optimizer', action='store_true')
    parser.add_argument('--lr', action='store_true')
    parser.add_argument('--wd', action='store_true')
    parser.add_argument('--batch_size', action='store_true')
    args = parser.parse_args()
    return args