import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject

import sys

Gst.init(None)

def create_stereo_source_bin(bin_name, left_uri, right_uri):
    """
    Simplified: we just create 1 source (e.g. left) for demo.
    Thực tế bạn có thể tự build 1 bin ghép left/right trước khi vào nvinfer.
    """
    bin = Gst.Bin.new(bin_name)

    src = Gst.ElementFactory.make("uridecodebin", f"{bin_name}_src")
    src.set_property("uri", left_uri)

    convert = Gst.ElementFactory.make("nvvideoconvert", f"{bin_name}_convert")
    capsfilter = Gst.ElementFactory.make("capsfilter", f"{bin_name}_caps")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
    )

    if not bin or not src or not convert or not capsfilter:
        print(f"Unable to create source elements for {bin_name}")
        return None

    bin.add(src)
    bin.add(convert)
    bin.add(capsfilter)

    # uridecodebin is dynamic pad, connect to convert
    def on_pad_added(decodebin, pad, data):
        sink_pad = convert.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    src.connect("pad-added", on_pad_added, None)

    convert.link(capsfilter)

    ghost_pad = Gst.GhostPad.new("src", capsfilter.get_static_pad("src"))
    bin.add_pad(ghost_pad)

    return bin

def main():
    if len(sys.argv) < 4:
        print("Usage: python hitnet_deepstream_multi.py <stereo0_left_uri> <stereo1_left_uri> <stereo2_left_uri>")
        sys.exit(1)

    left0_uri = sys.argv[1]
    left1_uri = sys.argv[2]
    left2_uri = sys.argv[3]

    pipeline = Gst.Pipeline.new("hitnet-multi-stereo-pipeline")

    # 3 stereo sources (demo chỉ dùng left URI)
    src0_bin = create_stereo_source_bin("stereo0", left0_uri, None)
    src1_bin = create_stereo_source_bin("stereo1", left1_uri, None)
    src2_bin = create_stereo_source_bin("stereo2", left2_uri, None)

    if not pipeline or not src0_bin or not src1_bin or not src2_bin:
        print("Failed to create pipeline or source bins")
        sys.exit(1)

    # streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", 3)
    streammux.set_property("batched-push-timeout", 40000)

    # nvinfer với HitNet
    pgie = Gst.ElementFactory.make("nvinfer", "primary-infer-engine")
    pgie.set_property("config-file-path",
        "/workspace/src/ct_uav_depth_package/hitnet/config_infer_hitnet.txt"
    )

    # converter + OSD để xem
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    sink = Gst.ElementFactory.make("nveglglessink", "video-sink")
    sink.set_property("sync", False)

    for elem in [streammux, pgie, nvvidconv, nvosd, sink]:
        if not elem:
            print("Unable to create element")
            sys.exit(1)

    pipeline.add(src0_bin)
    pipeline.add(src1_bin)
    pipeline.add(src2_bin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link source bins to streammux
    for i, bin in enumerate([src0_bin, src1_bin, src2_bin]):
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = bin.get_static_pad("src")
        if not sinkpad or not srcpad:
            print(f"Unable to get pads for stream {i}")
            sys.exit(1)
        srcpad.link(sinkpad)

    # Link rest of the pipeline
    if not Gst.Element.link(streammux, pgie):
        print("Could not link streammux to pgie")
        sys.exit(1)

    if not Gst.Element.link(pgie, nvvidconv):
        print("Could not link pgie to nvvidconv")
        sys.exit(1)

    if not Gst.Element.link(nvvidconv, nvosd):
        print("Could not link nvvidconv to nvosd")
        sys.exit(1)

    if not Gst.Element.link(nvosd, sink):
        print("Could not link nvosd to sink")
        sys.exit(1)

    # Bus
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error: %s: %s" % (err, debug))
            loop.quit()
        return True

    loop = GObject.MainLoop()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    print("Stopping pipeline")
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()