Files for openvinotoolkit/openvino#38727 (issue #38726).

repro_issue.py is the reproducer from the issue.

verify.py builds FQ1 then FQ2 with a direct link, a Transpose or a Reshape between them, per tensor and per channel ranges, output range equal to or different from the input range and 3, 16 and 256 levels. It compiles each model on CPU with f32 precision and compares the result with Model.evaluate, which runs the reference implementation without graph transformations.

Both were run on arm64 macOS with OpenVINO built from 62b19dcfba6, first with the pass source from upstream master and then with the PR applied.
