import util
import models
import pandas as pd
import torch

if __name__ == '__main__':
    print(util.get_root_dir())
    print(util.get_images())
    print(util.get_labels())
    print(util.get_emotions())

    # test = pd.read_csv(util.get_labels())
    # print(test.loc[test['image_name'] == 'image_2.jpeg']['text_corrected'].iloc[0])
    fuuuuckmebro = False

    models.train_bert(util.get_emotions(), overwrite=False)
    img_embeds, text_inputs = models.get_embeddings(util.get_images(), util.get_labels(), overwrite=fuuuuckmebro)
    print(img_embeds.shape, text_inputs)
    print('sentiment analysis???')
    models.get_bert_sentiment(overwrite=fuuuuckmebro, test_cap=100)

    cls = models.feed_VisualBERT(img_embeds, text_inputs, overwrite = fuuuuckmebro)
    
    models.feed_BLIP(util.get_images(), overwrite=True)
    util.calc_cosine_sims('image_5996.jpg',cls) # cosine sim on fused embeds
    #util.calc_cosine_sims(3,torch.load(util.get_root_dir() / 'data' / 'image_embeddings.pt').mean(dim=1)) # cosine sim on image embeds

    