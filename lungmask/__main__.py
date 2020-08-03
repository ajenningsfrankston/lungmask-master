import sys
import argparse
import logging
from lungmask import mask
from lungmask import utils
import os
import SimpleITK as sitk
import pkg_resources
import numpy as np


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='input', type=str, help='Path to the input image, can be a folder for dicoms')
    parser.add_argument('output', metavar='output', type=str, help='Directory for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str, choices=['R231' ,'LTRCLobes','LTRCLobes_R231','R231CovidWeb'], default='R231')
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1", action='store_true')
    parser.add_argument('--nopostprocess', help="Deactivates postprocessing (removal of unconnected components and hole filling", action='store_true')
    parser.add_argument('--noHU', help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images", action='store_true')
    parser.add_argument('--batchsize', type=int, help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.", default=20)

    argsin = sys.argv[1:]

    print("argsin ",argsin)

    args = parser.parse_args(argsin)

    print("input file ",args.input)

    batchsize = args.batchsize

    if args.cpu:
        batchsize = 1
    
    input_image = utils.get_input_image(args.input)

    logging.info(f'Load model')
    
    if args.modelname == 'LTRCLobes_R231':
        result = mask.apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU)
    else:
        model = mask.get_model(args.modeltype, args.modelname)
        result = mask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU)
        
    if args.noHU:
        file_ending = args.output.split('.')[-1]
        print(file_ending)
        if file_ending in ['jpg','jpeg','png']:
            result = (result/(result.max())*255).astype(np.uint8)
        result = result[0]
             
    result_out= sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)
    logging.info(f'Save result to: {args.output}')
    sys.exit(sitk.WriteImage(result_out, args.output))


if __name__ == "__main__":
    print('called as script')
    main()
