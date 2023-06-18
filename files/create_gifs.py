import os
from utils import create_gif_from_images

def create_gif(model_name, image_name_pattern, output_gif_name, model_path, num_epochs, cwd, index=None, crop_image=False):
    font_path = os.path.join(cwd, "Arial_Bold.ttf")
    create_gif_from_images(image_folder=model_path,
                           image_name_pattern=image_name_pattern,
                           output_gif_name=output_gif_name,
                           image_index=index if index is not None else 0,
                           img_height=416,
                           img_width=416,
                           font_path=font_path,
                           num_epochs=num_epochs,
                           crop_image=crop_image)

if __name__ == "__main__":
    NUM_EPOCHS = 200
    model_folder = "Experiment1"
    model_name = "UNet_NL"
    cwd = os.getcwd()
    model_path = os.path.join(os.path.dirname(cwd),"models", model_folder)

    # create GIFs
    for index in [2, 5, 8, 11, 14]:
        image_name_pattern = "{}_Epoch(\d{{1,3}})_pred.png".format(model_name)
        output_gif_name = model_name + "_pred" + "_GIF" + str(index) + ".gif"
        create_gif(model_name, image_name_pattern, output_gif_name, model_path, NUM_EPOCHS, cwd, index=index,
                   crop_image=True)

    image_name_pattern = "{}_Epoch(\d{{1,3}})_aggregated".format(model_name)
    output_gif_name = model_name + "_fm_aggregated" + "_GIF.gif"
    create_gif(model_name, image_name_pattern, output_gif_name, model_path, NUM_EPOCHS, cwd, crop_image=False)

    image_name_pattern = "{}_Epoch(\d{{1,3}})_GradCAM".format(model_name)
    output_gif_name = model_name + "GradCAM++" + "_GIF.gif"
    create_gif(model_name, image_name_pattern, output_gif_name, model_path, NUM_EPOCHS, cwd, crop_image=False)