import nltk
import spacy
import numpy as np
import pandas as pd
import copy as cp

sentence = "Philip Lombard, summing up the girl opposite in a mere flash of his quick moving eyes thought to himself: “Quite attractive - a bit schoolmistressy perhaps...” A cool customer, he should imagine - and one who could hold her own - in love or war. He’d rather like to take her on... He frowned. No, cut out all that kind of stuff. This was business. He’d got to keep his mind on the job. What exactly was up, he wondered? That little Jew had been damned mysterious. “Take it or leave it, Captain Lombard.” He had said thoughtfully: “A hundred guineas, eh?” He had said it in a casual way as though a hundred guineas was nothing to him. A hundred guineas when he was literally down to his last square meal! He had fancied, though, that the little Jew had not been deceived - that was the damnable part about Jews, you couldn’t deceive them about money - they knew! He had said in the same casual tone: “And you can’t give me any further information?” Mr. Isaac Morris had shaken his little bald head very positively. “No, Captain Lombard, the matter rests there. It is understood by my client that your reputation is that of a good man in a tight place. I am empowered to hand you one hundred guineas in return for which you will travel to Sticklehaven, Devon. The nearest station is Oakbridge, you will be met there and motored to Sticklehaven where a motor launch will convey you to Indian Island. There you will hold yourself at the disposal of my client.” Lombard had said abruptly: “For how long?” “Not longer than a week at most.” Fingering his small moustache, Captain Lombard said: “You understand I can’t undertake anything - illegal?” He had darted a very sharp glance at the other as he had spoken. There had been a very faint smile on the thick Semitic lips of Mr. Morris as he answered gravely: “If anything illegal is proposed, you will, of course, be at perfect liberty to withdraw.” Damn the smooth little brute, he had smiled!  It was as though he knew very well that in Lombard’s past actions legality had not always been a sine qua non... Lombard’s own lips parted in a grin."
words = sentence.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print(bag_of_words)

# Annotated words:
# nltk.download('popular')
# Using natural language toolkit
print("Using natural language toolkit:")
pos_tags = nltk.pos_tag(sentence.split())
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("Using spacy to get parts of speech tags.")
nlp = spacy.load('en_core_web_sm')
pos_tags_2 = [ (word, word.tag_,  word.pos_) for word in nlp(sentence)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)


