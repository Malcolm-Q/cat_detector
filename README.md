# cat_detector
Simple computer vision solution to detect my cat and turn on the vacuum to save my plants.

This project uses opencv, Tensorflow, efficientnetB0, [and this brilliant community library for Kasa smart home devices.](https://github.com/python-kasa/python-kasa)

After months of trying to make our plants harder for our cat to reach (unsuccessfully) I've come up with a foolproof solution. The best defense really is a strong offense.

Here is a video of it in action:

<video controls>
  <source src="cat_defense.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Note that the script uses the head of B0 as it contains a class for a tabby cat. My cat has a very distinct tabby coat and after some testing it can classify her reliably.

I set the model to run once per second to conserve resources but it is capable of running much faster.

On line 30 I do a simple and fast numpy slice crop on the image to the original input size of B0 (224x224) but you will likely need to do some other processing method.