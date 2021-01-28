import argparse
from utils.layers import *
from model import *
from utils.parse_config import *

ONNX_EXPORT = True


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', img_dim=(1024, 1024), ONNX_EXPORT=True):

    # Initialize model
    img_dim = tuple(img_dim) if type(img_dim) != tuple else img_dim
    model = Darknet(cfg, img_size=img_dim)

    # Load weights and save
    if weights.endswith('.weights'):  # darknet format
        load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        target = weights.rsplit('.', 1)[0] + '.pt'
        torch.save(chkpt, target)
        print("Success: converted '%s' to '%s'" % (weights, target))

    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_dim)  # (1, 3, 320, 192)
        f = weights.replace('.' + weights.split('.')[-1], '.onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])
        print(torch.__version__)

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    else:
        print('Error: extension not supported.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('img_dim', nargs='+', type=int, default=(1024, 1024))

    opt = parser.parse_args()
    convert(opt.cfg, opt.weights, opt.img_dim)