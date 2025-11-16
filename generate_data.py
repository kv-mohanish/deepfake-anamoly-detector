import tensorflow_datasets as tfds

# The `download_and_prepare` function will look in the manual folder
# you created and generate the dataset splits from the zip file.
# It will only download from the internet if the file isn't found locally.
builder = tfds.builder('celeb_a')
builder.download_and_prepare()

print("CelebA dataset generated successfully!")
