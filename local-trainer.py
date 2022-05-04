import boto3
import botocore
from botocore.config import Config
from zipfile import ZipFile
from keras.preprocessing import image
from keras.layers import Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense


BS = 32
TS = (24, 24)


def main():
    connect() # this process will fail without aws access codes and secrets alongwith right configuration to bucket and file
    downloader()
    machine_learning()



def connect():
    print("Connecting to S3")
    BUCKET_NAME = 'godseyeimagedatabase' # replace with your bucket name
    KEY = 'data.zip' # replace with your object key
    s3 = boto3.resource('s3',
                        aws_access_key_id='###',  # aws access key goes here
                        aws_secret_access_key="###") # aws secret goes here
    my_config = Config(
        region_name = 'us-east-1',
        signature_version = 'v4',
        retries = {
            'max_attempts': 10,
            'mode': 'standard'
        }
    )

    client = boto3.client('kinesis', config=my_config)

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, 'data.zip')
        print("download success")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def downloader():
    file_name = "data.zip"
    with ZipFile(file_name, 'r') as zip:
        zip.extractall()
    print("extract success")

def machine_learning():
    def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), brightness_range=[0.2, 1.2],
                  horizontal_flip=True,
                  vertical_flip=True, rotation_range=30, fill_mode='nearest', width_shift_range=0.2,
                  height_shift_range=0.2,
                  shuffle=True, batch_size=1, target_size=(24, 24), class_mode='categorical'):
        return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                       class_mode=class_mode, target_size=target_size)
    train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
    valid_batch = generator('data/test', shuffle=True, batch_size=BS, target_size=TS)
    SPE = len(train_batch.classes) // BS
    VS = len(valid_batch.classes) // BS


    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        # 32 convolution filters used each of size 3x3
        # again
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        # 64 convolution filters used each of size 3x3
        # choose the best features via pooling
        # randomly turn neurons on and off to improve convergence
        Dropout(0.25),
        # flatten since too many dimensions, we only want a classification output
        Flatten(),
        # fully connected to get all relevant data
        Dense(128, activation='relu'),
        # one more dropout for convergence' sake :)
        Dropout(0.5),
        # output a softmax to squash the matrix into output probabilities
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE,
                                  validation_steps=VS)

    model.save('models/cnnCat2.h5', overwrite=True)

main()