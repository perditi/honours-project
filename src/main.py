import util
import models
import pandas as pd

if __name__ == '__main__':
    print(util.get_root_dir())
    print(util.get_images())
    print(util.get_labels())

    # test = pd.read_csv(util.get_labels())
    # print(test.loc[test['image_name'] == 'image_2.jpeg']['text_corrected'].iloc[0])


    img_embeds, text_embeds = models.get_embeddings(util.get_images(), util.get_labels(), overwrite=True, test_cap=10)
    print(img_embeds.shape, text_embeds.shape)