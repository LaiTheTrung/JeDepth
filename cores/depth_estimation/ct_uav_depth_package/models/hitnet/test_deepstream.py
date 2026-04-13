#!/usr/bin/env python3
"""Test DeepStream inference pipeline for HitNet.

This script builds a simple GStreamer + DeepStream pipeline:
- Reads a stereo input source (e.g. video file or camera)
- Uses nvstreammux + nvinfer with HitNet TensorRT engine
- Uses custom parser nvdsinfer_custom_impl_hitnet.so
- Displays the output with nveglglessink

This is meant as a basic sanity test that the DeepStream
HitNet integration is working, not a full application.
"""

import argparse
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject


def print_info(msg: str):
    print(f"[INFO] {msg}")


def print_error(msg: str):
    print(f"[ERROR] {msg}")


def on_bus_message(bus, message, loop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print_info("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print_error(f"{err}: {debug}")
        loop.quit()
    return True


def build_pipeline(args):
    """Build DeepStream pipeline for HitNet inference."""
    Gst.init(None)

    pipeline = Gst.Pipeline.new("hitnet-deepstream-pipeline")
    if not pipeline:
        raise RuntimeError("Failed to create pipeline")

    # Source element: use filesrc + decodebin for a generic file
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        raise RuntimeError("Could not create filesrc")
    source.set_property("location", args.input)

    decodebin = Gst.ElementFactory.make("decodebin", "decoder")
    if not decodebin:
        raise RuntimeError("Could not create decodebin")

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        raise RuntimeError("Could not create nvstreammux")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batched-push-timeout", 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-infer-engine")
    if not pgie:
        raise RuntimeError("Could not create nvinfer")
    pgie.set_property("config-file-path", args.config)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    if not nvvidconv:
        raise RuntimeError("Could not create nvvideoconvert")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        raise RuntimeError("Could not create nvdsosd")

    sink = Gst.ElementFactory.make("nveglglessink", "video-sink")
    if not sink:
        raise RuntimeError("Could not create nveglglessink")
    sink.set_property("sync", False)

    pipeline.add(source)
    pipeline.add(decodebin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link source -> decodebin (dynamic pads)
    if not source.link(decodebin):
        raise RuntimeError("Failed to link source to decodebin")

    # Link static part: streammux -> pgie -> nvvidconv -> nvosd -> sink
    if not Gst.Element.link(streammux, pgie):
        raise RuntimeError("Failed to link streammux to nvinfer")
    if not Gst.Element.link(pgie, nvvidconv):
        raise RuntimeError("Failed to link nvinfer to nvvideoconvert")
    if not Gst.Element.link(nvvidconv, nvosd):
        raise RuntimeError("Failed to link nvvideoconvert to nvdsosd")
    if not Gst.Element.link(nvosd, sink):
        raise RuntimeError("Failed to link nvdsosd to sink")

    # Connect decodebin to streammux dynamically
    def on_pad_added(decodebin, pad, user_data):
        caps = pad.get_current_caps()
        name = caps.to_string() if caps else ""
        print_info(f"Decodebin pad-added with caps: {name}")

        if not name.startswith("video/"):
            return

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            print_error("Failed to get sink_0 pad from streammux")
            return

        if sinkpad.is_linked():
            print_error("streammux sink_0 already linked")
            return

        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            print_error("Failed to link decodebin to streammux")
        else:
            print_info("Linked decodebin to streammux")

    decodebin.connect("pad-added", on_pad_added, None)

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Test DeepStream HitNet inference pipeline")
    parser.add_argument("--input", type=str, required=True,
                        help="Input video file or stream URI (already encoded stereo feed)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to nvinfer config for HitNet (using custom parser)")
    parser.add_argument("--width", type=int, default=320,
                        help="Muxer width (default: 320)")
    parser.add_argument("--height", type=int, default=240,
                        help="Muxer height (default: 240)")

    args = parser.parse_args()

    Gst.init(None)
    loop = GObject.MainLoop()

    try:
        pipeline = build_pipeline(args)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    print_info("Starting DeepStream HitNet pipeline ...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print_info("Interrupted by user")

    print_info("Stopping pipeline ...")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
