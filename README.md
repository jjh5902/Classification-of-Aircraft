# Classification of Aircraft

## requirement
* tensorflow-gpu(1.2.x)
* python 3.5.x (I did not do it in anaconda.)

## Download

### weight and tfrecord file
This is the tfrecord file needed for source code testing only. If you only want to test it, you can get the file below. You can unpack it into a folder like run.py.

* Windows

```
https://drive.google.com/file/d/1kH2nsj2gzinlOKyEOMEBYru2l1_ySo0B/view?usp=sharing
```

* Linux

```
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1kH2nsj2gzinlOKyEOMEBYru2l1_ySo0B" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > dataset.tar.gz
```
### original image file
If you are interested in the original image file, you can get the following file.

* Windows

```
https://drive.google.com/open?id=1bRH9O8filwej0Rz6UG6BwmaHXuCm1UK7
```

* Linux

```
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1bRH9O8filwej0Rz6UG6BwmaHXuCm1UK7" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > data.zip
```
