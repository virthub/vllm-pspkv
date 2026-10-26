while ! test -f '/root/zhang/pspkv-format-compare-20261023/serving-finished.txt'; do sleep 5; done
docker run --name pspkv-format-20261023-control --platform linux/arm64 --entrypoint /bin/bash \
--device /dev/davinci4 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
-v /root/zhang/vllm-pspkv:/workspace/pspkv-src:ro \
-v /root/zhang/pspkv-native-build:/workspace/pspkv-native-build:ro \
-v /root/zhang/pspkv-format-compare-20261023:/comparison -v /usr/local/dcmi:/usr/local/dcmi:ro \
-v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-e PYTHONPATH=/workspace/pspkv-src:/vllm-workspace/vllm-ascend \
-e PSPKV_ASCEND_NATIVE_LIB=/workspace/pspkv-native-build/libpspkv_ascend_native.so \
quay.io/ascend/vllm-ascend:v0.20.2rc1 -lc 'bash /comparison/run-control.sh'
date -u +%FT%TZ > '/root/zhang/pspkv-format-compare-20261023/control-finished.txt'