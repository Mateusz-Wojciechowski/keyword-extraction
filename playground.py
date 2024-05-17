import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import OrderedDict

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def create_phrase_list(words, stop_words, punctuation, min_length, max_length):
    phrases_list = []
    phrase = []
    for word in words:
        if word.lower() in stop_words or word in punctuation:
            if min_length <= len(phrase) <= max_length:
                phrases_list.append(' '.join(phrase))
            phrase = []
        else:
            phrase.append(word)
    if min_length <= len(phrase) <= max_length:
        phrases_list.append(' '.join(phrase))
    return phrases_list


def create_word_dict(filtered_words):
    word_dict = {}
    for word in filtered_words:
        word_dict[word] = {word_: 0 for word_ in filtered_words}
    return word_dict


def update_word_dict(phrases_list, word_dict):
    for phrase in phrases_list:
        splited_phrase = phrase.split()
        for word in splited_phrase:
            for word1 in splited_phrase:
                word_dict[word][word1] += 1


def calculate_coefs(word_dict):
    word_coefs = {}
    for key, value in word_dict.items():
        degree_val = sum(value.values())
        if value[key] > 0:
            word_coefs[key] = degree_val / value[key]
        else:
            word_coefs[key] = 0
    return word_coefs


def calculate_phrases_scores(phrases_list, word_coefs):
    phrase_dict = {}
    for phrase in phrases_list:
        phrase_dict[phrase] = sum(word_coefs.get(word, 0) for word in phrase.split())
    return phrase_dict


def sort_result_dict(phrase_dict):
    return OrderedDict(sorted(phrase_dict.items(), key=lambda item: item[1], reverse=True))


def get_top_n_key_phrases(num_phrases, results_ordered):
    phrase_list = []
    for i, (keyword, _) in enumerate(results_ordered.items()):
        if i >= num_phrases:
            break
        phrase_list.append(keyword)
    return phrase_list


def rake_extraction(min_length, max_length, text, num_phrases):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    phrases_list = create_phrase_list(words, stop_words, string.punctuation, min_length, max_length)
    word_dict = create_word_dict(filtered_words)
    update_word_dict(phrases_list, word_dict)
    word_coefs = calculate_coefs(word_dict)
    phrase_dict = calculate_phrases_scores(phrases_list, word_coefs)
    results_ordered = sort_result_dict(phrase_dict)
    top_n_phrases = get_top_n_key_phrases(num_phrases, results_ordered)
    return top_n_phrases


text = """donald trump continued his march toward the gop nomination at the michigan republican party convention on saturday sweeping all 39 delegate related a farright u youth group is ramping up it movement to back election denier the delegate awarded will fuel the former president ahead of tuesday 5 march when 15 state will hold primary and trump nomination could be all but decided the michigan state party delegate met on saturday at the sprawling amway plaza hotel in grand rapid huddling in 13 separate meeting room representing the state 13 congressional district their nearuniform support for trump at the convention eclipsed the support he earned in the primary when former un ambassador nikki haley garnered about 26 of the vote she did not win any delegate awarded on saturday for the republican national convention in milwaukee where the party in july will officially nominate a candidate for the november presidential election the michigan republican party process for awarding delegate to the national committee wa complicated this year the democraticcontrolled state legislature decided to hold the presidential primary early this prompted the state republican party to create a hybrid model holding a primary on 27 february and a convention four day later to remain in compliance with the national party rule the convention on saturday at time took on the tone of a campaign rally president trump im going to help you win michigan exclaimed bernadette smith a michigan republican party activist running to be michigan republican national convention committeewoman during a speech at the convention saturday im from detroit i wa raised in detroit said smith to cheer detroit is red they just dont know it yet but if delegate found common cause today it wa only in their unyielding support for trump the michigan republican party ha been split for month over interpersonal feud in the county chapter the role of christian nationalism in the party at large and question about how to salvage the party from financial collapse the division fomenting in the party broke into the open this year in a leadership dispute when a group opposing the former michigan gop chair kristina karamo voted to oust her in january the republican national committee in february recognized pete hoekstra a close trump ally whom karamos opponent elected to chair the party a the rightful leader of the michigan gop karamo and her ally refused to accept defeat vowing to hold a separate convention in detroit which fell apart only after a judge ruled on tuesday that karamo had been properly removed from her seat and forbade her from using official michigan gop social medium account or accessing it finance before she wa elected last year to chair the michigan republican party karamo made a name for herself a a vocal proponent of trump false claim of widespread voter fraud during the 2020 election in michigan karamo went on to run for michigan secretary of state the office overseeing election in the state in 2022 she lost by 14 point but never conceded karamo who ha developed a reputation for floating outlandish conspiracy theory and who embrace christian nationalism ha referred to the split within the party a a form of spiritual warfare and her political opponent a demonic rhetoric embraced by section of the growing rightwing pentecostal movement in the u republican in the party were willing to look past the stranger aspect of their eccentric chair but when she failed to salvage the party struggling finance even splurging on a 100000 fee to bring jim caviezel the qanonaffiliated star of the passion of christ to speak at the mackinac republican leadership conference many grew frustrated with her but she ha retained loyalist in the party many of whom planned to attend karamos alternate gop convention in detroit before it wa canceled without a convention of their own some supporter of the former chair changed course at the last minute opting instead to attend the official one and lobbying mostly successfully for recognition in grand rapid others abandoned the convention entirely choosing to stay home or decamp to various alternative meeting held around the state on the same day republican party leader from the 1st congressional district which contains 15 county in the upper peninsula informed member friday that their district would be caucusing separately amid concern that the official convention would not accept their delegate the newly declared administration of the michigan republican party appears to be inviting dissent and disregarding rule with the consent of their michigan republican party ally said district chair daire rendon in a statement we will not play that game by falling into their confusing messaging and backtracking daire rendon did u a favor said tom stilling a michigan gop activist and former chair of the antrim county republican party which is in the first congressional district all the extremist were out there and the fear wa that they would be here without many of their delegate the 1st congressional caucus room sat mostly empty but rift in the michigan gop cut deeper than the crisis of leadership that the party ha faced this year often playing out at the county level in the republican party of hillsdale for example a small and conservative county in southern michigan party activist have been embroiled in a parallel dispute for year one thats been fought between the party and a faction of the party dubbed the america first republican party a judge in april 2023 ruled that the america first faction were not the legal leader of the party and in january found numerous activist including karamo in contempt of court for failing to recognize the ruling party activist in the 5th congressional district which stretch across the south of the state and represents hillsdale county tried to tamp down that dispute on saturday we all want to prevent a revolt said suzy avery a prominent michigan conservative who sits on the board of the michigan republican party trust and who resides in hillsdale avery who caucus with the hillsdale county republican party helped broker a deal with the america first activist granting that faction nine of the county party 13 delegate a similar split grew last year in kalamazoo county leading to a physical altercation during a state gop meeting last year leader in the michigan republican party downplayed intraparty tension on saturday viewing their ability to shepherd delegate through the saturday convention a a success today wa relatively uneventful it exciting that we can maybe move on said vance patrick the chair of the oakland county republican party the largest chapter of the michigan gop the crazy part about all this is everyone here is for trump"""
print(rake_extraction(1, 1, text, 10))
