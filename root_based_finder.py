#! /usr/bin/python
# encoding: utf-8

from itertools import chain

alphabet = set([u'ཀ', u'ཁ', u'ག', u'ང',
                    u'ཅ', u'ཆ', u'ཇ', u'ཉ',
                    u'ཏ', u'ཐ', u'ད', u'ན',
                    u'པ', u'ཕ', u'བ', u'མ',
                    u'ཙ', u'ཚ', u'ཛ', u'ཝ',
                    u'ཞ', u'ཟ', u'འ', u'ཡ',
                    u'ར', u'ལ', u'ཤ', u'ས',
                    u'ཧ', u'ཨ'])

pref = set([u"ག", u"ད", u"བ", u"མ", u"འ"])

head_letter = set([u"ར", u"ལ", u"ས"])

root_only = frozenset((u'ཀ', u'ཁ', u'ཅ', u'ཆ', u'ཇ', u'ཉ', u'ཏ', u'ཐ', u'པ', u'ཕ',
                     u'ཙ', u'ཚ', u'ཛ', u'ཝ', u'ཞ', u'ཟ', u'ཡ', u'ཤ',))

subcons = frozenset((u'ྐ', u'ྑ', u'ྒ', u'ྔ', u'ྕ', u'ྖ', u'ྗ', u'ྙ',
                                        u"ྚ", u'ྟ', u'ྠ', u'ྡ', u"ྜ", u'ྣ', u'ྤ', u'ྦ',
                                        u'ྥ', u'ྨ', u'ྩ', u'ྪ', u'ྫ', u'ྯ', u'ྮ', u'ྴ', u'ྷ', u'ྻ', u'ྼ', u'ྶ'))

subjoined = frozenset((u'ྱ', u'ྲ', u'ླ', u'ྭ')) # wazur is being treated as an official member, for now at least

suffixes = set([u'ག', u'ང', u'ད', u'ན', u'བ', u'མ', u'འ', u'ར', u'ལ', u'ས'])
second_suffix = set([u'ས', u'ད'])
vowels = set([u'ི', u'ུ', u'ེ', u'ོ'])

retroflex = frozenset((u'ཊ',u'ཋ',u'ཌ',u'ཎ',u'ཥ',
                      u'ྚ', u'ྛ', u'ྜ', u'ྞ', u'ྵ'))


twelve_ra_mgo = set([u'རྐ', u'རྒ', u'རྔ', u'རྗ', u'རྙ', u'རྟ', u'རྡ', u'རྣ',
                             u'རྦ', u'རྨ', u'རྩ', u'རྫ'])

ten_la_mgo = set([u'ལྐ', u'ལྒ', u'ལྔ', u'ལྕ', u'ལྗ', u'ལྟ', u'ལྡ', u'ལྤ',
                        u'ལྦ', u'ལྷ'])

eleven_sa_mgo = set([u'སྐ', u'སྒ', u'སྔ', u'སྙ', u'སྟ', u'སྡ', u'སྣ', u'སྤ',
                                u'སྦ', u'སྨ', u'སྩ'])

wazur_sub = set([u'ཀྭ',u'ཁྭ',u'གྭ',u'ཅྭ',u'ཉྭ',u'ཏྭ',u'དྭ',u'ཙྭ',u'ཚྭ',u'ཞྭ',u'ཟྭ',u'རྭ',
                 u'ལྭ',u'ཤྭ',u'སྭ',u'ཧྭ',u'གྲྭ', u'དྲྭ'])  #  everything after and including གྲྭ added by me

# 'dogs can combinations
seven_ya_tags = set([u'ཀྱ', u'ཁྱ', u'གྱ', u'པྱ', u'ཕྱ', u'བྱ', u'མྱ'])
twelve_ra_tags = set([u'ཀྲ', u'ཁྲ', u'གྲ', u'ཏྲ', u'ཐྲ', u'དྲ', u'པྲ', u'ཕྲ', u'བྲ', 
                  u'མྲ', u'ཧྲ', u'སྲ'])
six_la_tags = set([u'ཀླ', u'གླ', u'བླ', u'ཟླ', u'རླ', u'སླ'])

# three tiered stacks
ya_tags_stack = set([u'རྐྱ', u'རྒྱ', u'རྨྱ', u'སྐྱ', u'སྒྱ', u'སྤྱ', u'སྦྱ', u'སྨྱ'])
ra_tags_stack = set([u'སྐྲ', u'སྒྲ', u'སྣྲ', u'སྤྲ', u'སྦྲ', u'སྨྲ'])

legal_ga_prefix = frozenset([u'གཅ', u'གཉ', u'གཏ', u'གད', u'གན', u'གཙ', u'གཞ', 
                             u'གཟ', u'གཡ', u'གཤ', u'གས',])

legal_da_prefix = frozenset([u'དཀ', u'དཀྱ', u'དཀྲ', u'དག', u'དགྱ', u'དགྲ', u'དང', 
                            u'དཔ', u'དཔྱ', u'དཔྲ', u'དབ', u'དབྱ', u'དབྲ', u'དམ', 
                            u'དམྱ',])

legal_ba_prefix = frozenset([u'བཀ', u'བཀྱ', u'བཀྲ', u'བརྐ', u'བསྐ', u'བརྐྱ', u'བསྐྱ', 
                             u'བསྐྲ', u'བག', u'བགྱ', u'བརྒ', u'བསྒ', u'བརྒྱ', u'བསྒྱ', 
                             u'བསྒྲ', u'བརྔ', u'བསྔ', u'བཅ', u'བརྗ', u'བརྙ', u'བསྙ', 
                             u'བཏ', u'བརྟ', u'བལྟ', u'བསྟ', u'བད', u'བརྡ', u'བལྡ', 
                             u'བསྡ', u'བརྣ', u'བསྣ', u'བཙ', u'བརྩ', u'བསྩ', u'བརྫ', 
                             u'བཞ', u'བཟ', u'བཟླ', u'བརླ', u'བཤ', u'བས', u'བསྲ', 
                             u'བསླ', u'བགྲ'])

legal_ma_prefix = frozenset([u'མཁ', u'མཁྱ', u'མཁྲ', u'མག', u'མགྱ', u'མགྲ', u'མང',
                              u'མཆ', u'མཇ', u'མཉ', u'མཐ', u'མད', u'མན', u'མཚ', 
                              u'མཛ',])

legal_a_prefix = frozenset([u'འཁ',u'འཁྱ',u'འཁྲ',u'འག',u'འགྱ',u'འགྲ',u'འཆ',
                            u'འཇ',u'འཐ',u'འད',u'འདྲ',u'འཕ',u'འཕྱ',u'འཕྲ',
                            u'འབ',u'འབྱ',u'འབྲ',u'འཚ',u'འཛ',])

all_legal_prefix = (legal_ga_prefix.union(legal_da_prefix).union(legal_ma_prefix).
                    union(legal_ba_prefix).union(legal_a_prefix))

amb1 = (u'བགས', u'མངས')
amb2 = (u'དགས', u'འགས', u'དབས', u'དམས')


letters = (u'ཀ',u'ཁ',u'ག',u'གྷ',u'ང',u'ཅ',u'ཆ',u'ཇ',u'ཉ',u'ཊ',u'ཋ',u'ཌ',u'ཌྷ',u'ཎ',u'ཏ',
           u'ཐ',u'ད',u'དྷ',u'ན',u'པ',u'ཕ',u'བ',u'བྷ',u'མ',u'ཙ',u'ཚ',u'ཛ',u'ཛྷ',u'ཝ',
           u'ཞ',u'ཟ',u'འ',u'ཡ',u'ར',u'ལ',u'ཤ',u'ཥ',u'ས',u'ཧ',u'ཨ',u'ཀྵ',u'ཪ',u'ཫ',u'ཬ',)

all_stacks = frozenset([i for i in chain(twelve_ra_mgo ,ten_la_mgo , eleven_sa_mgo ,
                       seven_ya_tags , twelve_ra_tags , six_la_tags ,
                       ya_tags_stack , ra_tags_stack , wazur_sub)])

# for achung endings, also consider adding u'འུའོ'
achung_endings = set([u'འི', u'འུ', u'འང', u'འམ', u'འོ', u'འུའི'])

sa_yang_endings = set([u'གས',u'ངས',u'བས',u'མས',])
da_yang_endings = set([u'ནད',u'རད',u'ལད',]) # Rare

valid_starts = all_stacks.union(all_legal_prefix).union(alphabet).union(wazur_sub)
valid_endings = suffixes.union(sa_yang_endings).union(achung_endings).union(da_yang_endings)

head = head_letter

hard_indicators = root_only.union(subjoined).union(vowels).union(subcons)

letters = (u'ཀ',u'ཁ',u'ག',u'གྷ',u'ང',u'ཅ',u'ཆ',u'ཇ',u'ཉ',u'ཊ',u'ཋ',u'ཌ',u'ཌྷ',u'ཎ',u'ཏ',
           u'ཐ',u'ད',u'དྷ',u'ན',u'པ',u'ཕ',u'བ',u'བྷ',u'མ',u'ཙ',u'ཚ',u'ཛ',u'ཛྷ',u'ཝ',
           u'ཞ',u'ཟ',u'འ',u'ཡ',u'ར',u'ལ',u'ཤ',u'ཥ',u'ས',u'ཧ',u'ཨ',u'ཀྵ',u'ཪ',u'ཫ',u'ཬ',)

subjoined_letters = (u'ྐ',u'ྑ',u'ྒ',u'ྒྷ',u'ྔ',u'ྕ',u'ྖ',u'ྗ',u'ྙ',u'ྚ',u'ྛ',u'ྜ',u'ྜྷ',
                      u'ྞ',u'ྟ',u'ྠ',u'ྡ',u'ྡྷ',u'ྣ',u'ྤ',u'ྥ',u'ྦ',u'ྦྷ',u'ྨ',u'ྩ',
                      u'ྪ',u'ྫ',u'ྫྷ',u'ྭ',u'ྮ',u'ྯ',u'ྰ',u'ྱ',u'ྲ',u'ླ',u'ྴ',u'ྵ',
                      u'ྶ',u'ྷ',u'ྸ',u'ྐྵ',u'ྺ',u'ྻ',u'ྼ',)

f_vowels = (u'\u0f71', u'\u0f72', u'\u0f73', u'\u0f74', u'\u0f75', u'\u0f76',
     u'\u0f77', u'\u0f78', u'\u0f79', u'\u0f7a', u'\u0f7b', u'\u0f7c', u'\u0f7d',
     u'\u0f80', u'\u0f81')

misc_word_parts = (u'ྃ', u'ཾ')

word_parts = letters + subjoined_letters + f_vowels + misc_word_parts

def start_end_ok(s, e):
#    if len(vowels.intersection(e)) > 0:
#        return False
    # Wazur in combinations like གྲྭ is not accounted for in main routine
    # This stems from fact wazur is regarded as a subjoined letter
    # even though actually it itself can be appended to a subjoined ltr
    if e and e[0] == u'ྭ' and s:
        s += u'ྭ'
        e = e.lstrip(u'ྭ')

    if s and s not in valid_starts:
        return False

    # we look at the beginning and ending letters of a word
    # and for the most part assume any vowels in between are behaving
    # as expected. This is not always a good assumption.
    # In this case, we check to make sure there is not more than one
    # consecutive vowel
    if e and e[0] in vowels and len(e) > 1:
        if e[1] in vowels:
            return False

    e = e.lstrip(u'ིེོུ')

    if e and e not in valid_endings:
        return False

    return True


def find_root_easy(ls):
    '''Hard indicators tell you exactly where root is'''
    root = -1
    for i, l in enumerate(ls):
        
        if len(ls) == 1 and l in alphabet:
            root = 0
            break
        
        elif l in subjoined:

            if not start_end_ok(ls[0:i+1],ls[i+1:]):
#                print start, end
                root = -1
            else: root =  i - 1
            break
        elif l in vowels and ls[i-1] == u'འ' and i-1 != 0:
#            if i-1 != 0:
#                root = -1
#            else:
#                if not start_end_ok(u'འ' , ls[i+1:]):
#                    root = -1
#                else:
#                    root = 0
            if not start_end_ok(ls[:i-1] , ls[i-1:]):
                root = -1
                break
            else:
                root = 0
            break
        elif l in vowels:
            if not start_end_ok(ls[0:i], ls[i:]):
                root = -1
                break
            else: root = i-1
            break
        elif l in root_only or l in subcons:
            try:
                if i+1 <= len(ls) - 1 and ls[i+1] in subjoined:
                    if not start_end_ok(ls[:min(i+2, len(ls))], ls[min(i+2, len(ls)):]):
#                        print ls[:min(i+2, len(ls))], ls[min(i+2, len(ls)):], 'ookl'
                        root = -1
                    else:
                        root = i


                elif not start_end_ok(ls[:min(i+1, len(ls))], ls[min(i+1, len(ls)):]):
#                    print ls[:min(i+1, len(ls))], ls[min(i+1, len(ls)):], 'skldfj'
                    root = -1

                else:
                    root = i
            except IndexError:
                if not start_end_ok(ls, ''):
                     root = -1
                else:
                    root = i
            break


    return root

def find_root_cons(ls):
    '''Find root among a string of non descript consonants'''

    root = -1

    if len(ls) == 1:
        if ls in alphabet:
            root = 0
        else: root = -1

    if len(ls) == 2:
        if ls[0] == ls[1]:
            root = -1
        elif ls[1] == u'འ':
            root = -1
        elif ls[1] not in suffixes:
            root = -1
        else:
            root = 0

    elif len(ls) == 3:
        if ls[-1] in suffixes:
            if (ls[-2:] in sa_yang_endings) or\
               (ls[-2:] in da_yang_endings):
                if ls in amb2: #ambiguous cases
                    root = 1
                else:
                    root = 0
            elif ls[1] == u'འ': # ex བའམ
                if not start_end_ok(ls[0], ls[1:]): root = -1
                else: root = 0
            else:
                if not start_end_ok(ls[0:2], ls[2]): root = -1
                else:
                    root = 1


    elif len(ls) == 4:
#        print 'landed here'
        root = 1
        if not start_end_ok(ls[0:2], ls[2:]): root = -1

    return root


def is_non_std(ls):
    '''Detect whether a group of letters is non standard. ls
    is assumed to be word chars only. i.e. numbers, symbols, etc
    should be removed before calling this function'''

    if not retroflex.isdisjoint(ls):
        return True

    elif not hard_indicators.isdisjoint(ls):
        root_ind = find_root_easy(ls)
        if root_ind not in (0,1,2):
            return True
        else:
            return False

    else:
        if not alphabet.issuperset(ls):
            pass

        root = find_root_cons(ls)
        if root == -1:
            return True
        else:
            return False


def get_root(ls):
    if not retroflex.isdisjoint(ls):
        return ls[0]

    elif not hard_indicators.isdisjoint(ls):
        root_ind = find_root_easy(ls)
        if root_ind not in (0,1,2):
            return ls[0]
        else:
            return ls[root_ind]

    else:
        if not alphabet.issuperset(ls):
#            print 'Warning!', ls
            pass

        root_ind = find_root_cons(ls)
        if root_ind == -1:
            return ls[0]
        else:
            return ls[root_ind]



if __name__ == '__main__':
#    print start_end_ok(u'བ', u'གས')
    samples = u'བཏགས སྒྲུབ པའི འོད སྤྲེའུའི མཏོན  གཏོ ནཔལཐ གཞན མཐའ མདའ བདག ལནག ཀྲ ཁྲ བའམ པའམ མཐའི རེའུ ལ པ བ ན ལྟ རྒྱཔ པོདེ བསྡིག མགྲོད བགྲོད པོའོ  བའི ཧཱུྃ'
    for s in samples.split():
        print is_non_std(s), s

    from termset import syllables
    print u'ཧཱུྃ' in syllables

