# FLRE
Towards thousands to one reference: can we trust the reference image for quality assessment?
![](https://github.com/ytian73/FLRE/blob/master/FLRE/framework.png)

## Environment
  * torch=1.13.0
  * Python=3.8.3

## Test
  * For FLRE+DeepWSD: <br>
    `python ./FLRE_deepwsd.py --ref='./images/I04.BMP' --dist='./images/i04_06_4.bmp'`
  * For FLRE+DISTS: <br>
    `python ./FLRE_dists.py --ref='./images/I04.BMP' --dist='./images/i04_06_4.bmp'`
  * For FLRE+LPIPS: <br>
    `python ./FLRE_lpips.py --ref='./images/I04.BMP' --dist='./images/i04_06_4.bmp'`
