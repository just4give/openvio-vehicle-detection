#!/usr/bin/env python
import cv2
import numpy as np
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import time
import logging as log
from tempimage import TempImage
#import boto3
import pygame
from openvino.inference_engine import IECore,IENetwork, IEPlugin

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to image file.",
                      required=False, type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-b", "--bucket", help="Required. AWS S3 bucket name.",
                      required=False, type=str)

    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Loading Inference Engine")
    ie = IECore()

    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)

    model_xml = args.model #'face-detection-retail-0004.xml'
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)


    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" " * 8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
                                                          versions[args.device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    n, c, h, w = net.inputs[input_blob].shape
    del net

    #s3client = boto3.resource('s3')
    expiresIn=5*24*3600 #expires recorded voice after 5 days

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    last_epoch = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    ret, frame = cap.read()
    log.info("To close the application, press 'CTRL+C'")
    

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()

        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start
            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            #log.info("Processing output blobs")
            
            #print(res)
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    #color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    color = (232, 35, 244)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

                    diff= (time.time()-last_epoch)
                    
                    log.info("Object detected {:.1f} s".format(round(obj[2] * 100, 1)))
                    if diff> 60:
                        last_epoch=time.time()
                        
                        t = TempImage()
                        cv2.imwrite(t.path, frame)
                        #s3client.meta.client.upload_file(t.path, args.bucket, t.key )
                        #log.info("upload image to {}".format(signedUrl))
                        play_mp3('alert.wav')
                        t.cleanup()



            inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        render_start = time.time()
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start
    

        
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def play_mp3(file):
        pygame.mixer.init()
        pygame.init()
        pygame.mixer.music.load(file)
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.event.set_allowed(pygame.USEREVENT)
        pygame.mixer.music.play()
        pygame.event.wait() # play() is asynchronous. This wait forces the speaking to be finished before closing
            
        while pygame.mixer.music.get_busy() == True:
            pass
        print("finished playing file", file)

if __name__ == '__main__':
    sys.exit(main() or 0)
