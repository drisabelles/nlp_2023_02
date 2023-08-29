import nltk
import spacy
import numpy as np
import pandas as pd
import copy as cp
import joblib

sentence = "E a viagem prosseguiu, mais lenta, mais arrastada, num silêncio grande. Ausente do companheiro, a cachorra Baleia tomou a frente do grupo. Arqueada, as costelas à mostra, corria ofegando, a língua fora da boca. E de quando em quando se detinha, esperando as pessoas, que se retardavam. Ainda na véspera eram seis viventes, contando com o papagaio. Coitado, morrera na areia do rio, onde haviam descansado, à beira de uma poça: a fome apertara demais os retirantes e por ali não existia sinal de comida. Baleia jantara os pés, a cabeça, os ossos do amigo, e não guardava lembranças disto. Agora, enquanto parava, dirigia as pupilas brilhantes aos objetos familiares, estranhava não ver sobre o baú de folha a gaiola pequena onde a ave se equilibrava mal. Fabiano também às vezes sentia falta dela, mas logo a recordação chegava. Tinha andado a procurar raízes, à toa: o resto de farinha acabara, não se ouvia um berro de rês perdida na catinga. Sinha Vitoria, queimando o assento no chão, as mãos cruzadas segurando os joelhos ossudos, pensava em acontecimentos antigos que não se relacionavam: festas de casamento, vaquejadas, novenas, tudo numa confusão só. Despertara-a um grito áspero, vira de perto a realidade do papagaio, que andava furioso, com os pés apalhetados, numa atitude ridícula. Resolvera de supetão aproveitá-lo como alimento e justificara-se declarando a si mesma que ele era mudo e inútil. Não podia deixar de ser mudo. Ordinariamente a família falava pouco. E depois daquele desastre viviam todos calados, raramente soltavam palavras curtas. O louro aboiava, tangendo um gado inexistente, e latia arremedando a cachorra. As manchas dos juazeiros tornaram a aparecer, Fabiano aligeirou o passo, esqueceu a fome, a canseira e os ferimentos. As alpercatas dele estavam gastas nos saltos, e a embira tinha-lhe aberto entre os dedos rachaduras muito dolorosas. Os calcanhares, duros como cascos, gretavam-se e sangravam. Num cotovelo do caminho avistou um canto de cerca, encheu-o a esperança de achar comida, sentiu desejo de cantar."
words = sentence.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print(bag_of_words)

# Annotated words:
# nltk.download('popular')
# Using natural language toolkit
print("Usando o natural language toolkit:")
# Use lang with ISO 639 code of the language
#pos_tags = nltk.pos_tag(sentence.split(), lang="pt") # not implemented.

# Reference to get the trained model:
# https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
trained_data_folder = 'python/introduction/data/'
portuguese_tagger = joblib.load(trained_data_folder+'POS_tagger_brill.pkl')
pos_tags = portuguese_tagger.tag(nltk.word_tokenize(sentence))
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("Usando o Spacy para se obter as partes do discurso.")
## https://spacy.io/models/pt
model_spacy = spacy.load('pt_core_news_sm')
pos_tags_2 = [ (word, word.pos_) for word in model_spacy(sentence)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)



