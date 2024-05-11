import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from functionalities import create_phrase_list, preprocess_text
from collections import OrderedDict

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

text = """
         Donald Trump continued his march toward the GOP nomination at the Michigan Republican party convention on Saturday, sweeping all 39 delegates.       Related:  A far-right US youth group is ramping up its movement to back election deniers       The delegates awarded will fuel the former president ahead of Tuesday, 5 March, when 15 states will hold primaries and Trump’s nomination could be all but decided. The Michigan state party delegates met on Saturday at the sprawling Amway Plaza Hotel in Grand Rapids, huddling in 13 separate meeting rooms representing the state’s 13 congressional districts.   Their near-uniform support for Trump at the convention eclipsed the support he earned in the primary, when former UN ambassador Nikki Haley garnered about 26% of the vote. She did not win any delegates awarded on Saturday for the Republican national convention in Milwaukee, where the party in July will officially nominate a candidate for the November presidential election.   The Michigan Republican party’s process for awarding delegates to the national committee was complicated this year: the Democratic-controlled state legislature decided to hold the presidential primaries early. This prompted the state Republican party to create a “hybrid” model, holding a primary on 27 February and a convention four days later to remain in compliance with the national party’s rules.   The convention on Saturday at times took on the tone of a campaign rally.   “President Trump, I’m going to help you win Michigan,” exclaimed Bernadette Smith, a Michigan Republican party activist running to be Michigan’s Republican national convention committeewoman, during a speech at the convention Saturday. “I’m from Detroit – I was raised in Detroit,” said Smith, to cheers. “Detroit is red, they just don’t know it yet.”   But if delegates found common cause today, it was only in their unyielding support for Trump. The Michigan Republican party has been split for months over interpersonal feuds in the county chapters, the role of Christian nationalism in the party at large and questions about how to salvage the party from financial collapse.   The divisions fomenting in the party broke into the open this year in a leadership dispute when a group opposing the former Michigan GOP chair, Kristina Karamo, voted to oust her in January. The Republican national committee in February recognized Pete Hoekstra, a close Trump ally whom Karamo’s opponents elected to chair the party, as the rightful leader of the Michigan GOP.   Karamo and her allies refused to accept defeat, vowing to hold a separate convention in Detroit – which fell apart only after a judge ruled on Tuesday that Karamo had been properly removed from her seat and forbade her from using official Michigan GOP social media accounts or accessing its finances.   Before she was elected last year to chair the Michigan Republican party, Karamo made a name for herself as a vocal proponent of Trump’s false claims of widespread voter fraud during the 2020 election in Michigan. Karamo went on to run for Michigan secretary of state, the office overseeing elections in the state, in 2022. She lost by 14 points but never conceded.   Karamo, who has developed a reputation for floating outlandish conspiracy theories and who embraces Christian nationalism, has referred to the split within the party as a form of “spiritual warfare” and her political opponents as “demonic” – rhetoric embraced by sections of the growing rightwing Pentecostal movement in the US.   Republicans in the party were willing to look past the stranger aspects of their eccentric chair, but when she failed to salvage the party’s struggling finances – even splurging on a $100,000 fee to bring Jim Caviezel, the QAnon-affiliated star of The Passion of Christ, to speak at the Mackinac Republican Leadership Conference – many grew frustrated with her.   But she has retained loyalists in the party, many of whom planned to attend Karamo’s alternate GOP convention in Detroit before it was canceled.   Without a convention of their own, some supporters of the former chair changed course at the last minute, opting instead to attend the official one and lobbying – mostly successfully – for recognition in Grand Rapids.   Others abandoned the convention entirely, choosing to stay home or decamp to various alternative meetings held around the state on the same day. Republican party leaders from the 1st congressional district, which contains 15 counties in the Upper Peninsula, informed members Friday that their district would be caucusing separately amid concerns that the official convention would not accept their delegates.   “The newly declared administration of [the Michigan Republican party] appears to be inviting dissent and disregarding rules with the consent of their Michigan Republican party allies,” said district chair Daire Rendon, in a statement. “We will not play that game by falling into their confusing messaging and backtracking.”   “Daire Rendon did us a favor,” said Tom Stilling, a Michigan GOP activist and former chair of the Antrim county Republican party, which is in the first congressional district. “All the extremists were out there, and the fear was that they would be here.” Without many of their delegates, the 1st congressional caucus room sat mostly empty.   But rifts in the Michigan GOP cut deeper than the crisis of leadership that the party has faced this year, often playing out at the county level.   In the Republican party of Hillsdale, for example, a small and conservative county in southern Michigan, party activists have been embroiled in a parallel dispute for years – one that’s been fought between the party and a faction of the party dubbed the America First Republican party. A judge in April 2023 ruled that the America First faction were not the legal leaders of the party and in January found numerous activists, including Karamo, in contempt of court for failing to recognize the ruling.   Party activists in the 5th congressional district, which stretches across the south of the state and represents Hillsdale county, tried to tamp down that dispute on Saturday.   “We all want to prevent a revolt,” said Suzy Avery, a prominent Michigan conservative who sits on the board of the Michigan Republican Party Trust and who resides in Hillsdale. Avery, who caucuses with the Hillsdale county Republican party, helped broker a deal with the America First activists, granting that faction nine of the county party’s 13 delegates.   A similar split grew last year in Kalamazoo county, leading to a physical altercation during a state GOP meeting last year.   Leaders in the Michigan Republican party downplayed intraparty tensions on Saturday, viewing their ability to shepherd delegates through the Saturday convention as a success.   “Today was relatively uneventful – it’s exciting that we can maybe move on,” said Vance Patrick, the chair of the Oakland county Republican party, the largest chapter of the Michigan GOP.   “The crazy part about all this is everyone here is for Trump.”
      """

words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]


def create_word_dict():
    word_dict = {}
    for word in filtered_words:
        word_dict[word] = {}
        for word_ in filtered_words:
            word_dict[word].update({word_: 0})

    return word_dict


def update_word_dict(phrases_list, word_dict):
    for phrase in phrases_list:
        splited_phrase = phrase.split()
        for word in splited_phrase:
            for word1 in splited_phrase:
                word_dict[word][word1] += 1


def calculate_coofs(word_dict):
    word_coofs = {}

    for key, value in word_dict.items():
        total_val = 0
        degree_val = 0
        for key_inner, value_inner in value.items():
            if key_inner == key:
                degree_val += value_inner

            total_val += value_inner

        word_coofs[key] = total_val/degree_val

    return word_coofs


def calculate_phrases_scores(phrases_list, word_coofs):
    phrase_dict = {}
    for phrase in phrases_list:
        total_score = 0
        splited_phrase = phrase.split()
        for word in splited_phrase:
            total_score += word_coofs[word]

        phrase_dict[phrase] = total_score

    return phrase_dict


def sort_result_dict(phrase_dict):
    sorted_word_score_desc = sorted(phrase_dict.items(), key=lambda item: item[1], reverse=True)
    ordered_word_score_desc = OrderedDict(sorted_word_score_desc)

    return ordered_word_score_desc


phrases_list = create_phrase_list(words, stop_words, string.punctuation)
word_dict = create_word_dict()
update_word_dict(phrases_list, word_dict)
word_coofs = calculate_coofs(word_dict)
phrase_dict = calculate_phrases_scores(phrases_list, word_coofs)
results_ordered = sort_result_dict(phrase_dict)


for key, value in results_ordered.items():
    print(f"Phrase: {key} Score: {value}")

