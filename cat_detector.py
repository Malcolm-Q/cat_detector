import tensorflow as tf
import numpy as np
from PIL import Image
from kasa import SmartPlug
import datetime, asyncio, cv2

async def main():
    # connect to smart plug through IP address (run 'kasa' in your console to scan for devices)
    p = SmartPlug("device_ip")
    await p.update()

    # Use B0 with the head. (class 281 is tabby)
    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
    )

    # read and process cam loop
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()

        # Here I'm slicing a 224x224 chunk out the photo. This is the input size for the model and luckily perfectly crops the area of interest.
        # If you need you can of course process this using Pillow or another tool instead.
        img = img[212:436,150:374]

        cv2.imshow('Cat Cam', img)
        # expand and feed.
        img = np.expand_dims(img, axis=0)
        result = model.predict(img,verbose=0)

        if cv2.waitKey(1) == 27: # kill if escape is hit
            break

        # Grab and format time for log and saved file name.
        now = datetime.datetime.now().time().strftime("%H:%M:%S")
        if np.argmax(result) == 281: # 281 is the class for tabby cat
            print(now + ": CAT          ***")

            # turn on vacuum for 1 second
            await p.turn_on()
            await asyncio.sleep(1)
            await p.turn_off()

            # Save the image so we can check it next morning.
            img = Image.fromarray(np.squeeze(img,axis=0))
            img.save("cat_"+now.replace(':',',')+'.jpg')
        else: print(now + ": no cat")

        # This runs quite fast so you can set a lower sleep if you like. Always good to be nice to your PC though.
        await asyncio.sleep(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())