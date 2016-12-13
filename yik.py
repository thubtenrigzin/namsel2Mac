#! /usr/bin/python
# encoding: utf-8
'''Useful sets of characters and symbols
'''

alphabet = (u'ཀ', u'ཁ', u'ག', u'ང', 
                    u'ཅ', u'ཆ', u'ཇ', u'ཉ', 
                    u'ཏ', u'ཐ', u'ད', u'ན', 
                    u'པ', u'ཕ', u'བ', u'མ', 
                    u'ཙ', u'ཚ', u'ཛ', u'ཝ', 
                    u'ཞ', u'ཟ', u'འ', u'ཡ', 
                    u'ར', u'ལ', u'ཤ', u'ས',
                    u'ཧ', u'ཨ')

prefixes = (u"ག", u"ད", u"བ", u"མ", u"འ")

head_letter = (u"ར", u"ལ", u"ས")

root_only = frozenset((u'ཀ', u'ཁ', u'ཅ', u'ཆ', u'ཇ', u'ཉ', u'ཏ', u'ཐ', u'པ', u'ཕ',
                     u'ཙ', u'ཚ', u'ཛ', u'ཝ', u'ཞ', u'ཟ', u'ཡ', u'ཤ',))
                     #  u'ས' is listed as one technically

subjoined_consonants = frozenset((u'ྐ', u'ྑ', u'ྒ', u'ྔ', u'ྕ', u'ྖ', u'ྗ', u'ྙ', 
                                        u"ྚ", u'ྟ', u'ྠ', u'ྡ', u"ྜ", u'ྣ', u'ྤ', u'ྦ',
                                        u'ྥ', u'ྨ', u'ྩ', u'ྪ', u'ྫ', u'ྯ', u'ྮ', u'ྴ', u'ྷ', u'ྻ', u'ྼ'))


subjoined = (u'ྱ', u'ྲ', u'ླ', u'ྭ') # wazur is being treated as an official member, for now at least 

suffixes = (u'ག', u'ང', u'ད', u'ན', u'བ', u'མ', u'འ', u'ར', u'ལ', u'ས')
second_suffix = (u'ས', u'ད')
vowels1 = (u'ི', u'ུ', u'ེ', u'ོ')
vowels2 = (u'ཨི', u'ཨུ', u'ཨེ', u'ཨོ')
punctuation = (u'་', u'༌', u'།', u'༎', u'༏', u'༑', u'༔', u'༴')

#~ standard_group = ''
#~ for l in (alphabet, subjoined_consonants, subjoined, vowels1, vowels2, punctuation):
    #~ standard_group += u''.join(l)

#special_chars = ()

# mgo can combinations
twelve_ra_mgo = (u'རྐ', u'རྒ', u'རྔ', u'རྗ', u'རྙ', u'རྟ', u'རྡ', u'རྣ',
                             u'རྦ', u'རྨ', u'རྩ', u'རྫ')
ten_la_mgo = (u'ལྐ', u'ལྒ', u'ལྔ', u'ལྕ', u'ལྗ', u'ལྟ', u'ལྡ', u'ལྤ',
                        u'ལྦ', u'ལྷ')
eleven_sa_mgo = (u'སྐ', u'སྒ', u'སྔ', u'སྙ', u'སྟ', u'སྡ', u'སྣ', u'སྤ',
                                u'སྦ', u'སྨ', u'སྩ')

# 'dogs can combinations
seven_ya_tags = (u'ཀྱ', u'ཁྱ', u'གྱ', u'པྱ', u'ཕྱ', u'བྱ', u'མྱ')
twelve_ra_tags = (u'ཀྲ', u'ཁྲ', u'གྲ', u'ཏྲ', u'ཐྲ', u'དྲ', u'པྲ', u'ཕྲ', u'བྲ', u'མྲ', u'ཧྲ', u'སྲ')
six_la_tags = (u'ཀླ', u'གླ', u'བླ', u'ཟླ', u'རླ', u'སླ')

# three tiered stacks
ya_tags_stack = (u'རྐྱ', u'རྒྱ', u'རྨྱ', u'སྐྱ', u'སྒྱ', u'སྤྱ', u'སྦྱ', u'སྨྱ')
ra_tags_stack = (u'སྐྲ', u'སྒྲ', u'སྣྲ', u'སྤྲ', u'སྦྲ', u'སྨྲ')

# grammar 
seven_la_don = (u'སུ', u'ར', u'རུ', u'དུ', u'ན', u'ལ', u'ཏུ')
grel_sgra = (u'གི', u'ཀྱི', u'གྱི', u'འི', u'ཡི')
byed_sgra = (u'གིས', u'ཀྱིས', u'གྱིས', u'འིས', u'ཡིས')
terminating_syllables = (u'གོ', u'ངོ', u'དོ', u'ནོ', u'བོ', u'མོ', u'འོ', u'རོ', u'ལོ', u'སོ')
rgyan_sdud = (u'ཀྱང',u'འང',u'ཡང')
num = (u'༡', u'༢', u'༣', u'༤', u'༥', u'༦', u'༧', u'༨', u'༩', u'༠')

# ambiguous cases
amb1 = (u'བགས', u'མངས')
amb2 = (u'དགས', u'འགས', u'དབས', u'དམས')

    
letters = (u'\u0f40', u'\u0f41', u'\u0f42', u'\u0f43', u'\u0f44', u'\u0f45',
    u'\u0f46', u'\u0f47', u'\u0f49', u'\u0f4a', u'\u0f4b', u'\u0f4c', u'\u0f4d',
    u'\u0f4e', u'\u0f4f', u'\u0f50', u'\u0f51', u'\u0f52', u'\u0f53', u'\u0f54',
    u'\u0f55', u'\u0f56', u'\u0f57', u'\u0f58', u'\u0f59', u'\u0f5a', u'\u0f5b',
    u'\u0f5c', u'\u0f5d', u'\u0f5e', u'\u0f5f', u'\u0f60', u'\u0f61', u'\u0f62', 
    u'\u0f63', u'\u0f64', u'\u0f65', u'\u0f66', u'\u0f67', u'\u0f68', u'\u0f69', 
    u'\u0f6a', u'\u0f6b', u'\u0f6c')

subjoined_letters = (u'\u0f90', u'\u0f91', u'\u0f92', u'\u0f93', u'\u0f94',
    u'\u0f95', u'\u0f96', u'\u0f97', u'\u0f99', u'\u0f9a', u'\u0f9b', u'\u0f9c',
    u'\u0f9d', u'\u0f9e', u'\u0f9f', u'\u0fa0', u'\u0fa1', u'\u0fa2', u'\u0fa3',
    u'\u0fa4', u'\u0fa5', u'\u0fa6', u'\u0fa7', u'\u0fa8', u'\u0fa9', u'\u0faa',
    u'\u0fab', u'\u0fac', u'\u0fad', u'\u0fae', u'\u0faf', u'\u0fb0', u'\u0fb1',
    u'\u0fb2', u'\u0fb3', u'\u0fb4', u'\u0fb5', u'\u0fb6', u'\u0fb7', u'\u0fb8',
    u'\u0fb9', u'\u0fba', u'\u0fbb', u'\u0fbc')


f_vowels = (u'\u0f71', u'\u0f72', u'\u0f73', u'\u0f74', u'\u0f75', u'\u0f76',
     u'\u0f77', u'\u0f78', u'\u0f79', u'\u0f7a', u'\u0f7b', u'\u0f7c', u'\u0f7d', 
     u'\u0f80', u'\u0f81')

signs = (u'\u0f1a', u'\u0f1b', u'\u0f1c', u'\u0f1d', u'\u0f1e', u'\u0f1f',
    u'\u0f3e', u'\u0f3f', u'\u0f7e', u'\u0f7f', u'\u0f82', u'\u0f83', u'\u0f86',
    u'\u0f87', u'\u0f88', u'\u0f89', u'\u0f8a', u'\u0f8b', u'\u0fc0', u'\u0fc1',
    u'\u0fc2', u'\u0fc3', u'\u0fce', u'\u0fcf') # does not include logotypes or astrological signs

marks = (u'\u0f01', u'\u0f02', u'\u0f03', u'\u0f04', u'\u0f05', u'\u0f06',
    u'\u0f07', u'\u0f08', u'\u0f09', u'\u0f0a', u'\u0f0b', u'\u0f0c', u'\u0f0d',
    u'\u0f0e', u'\u0f0f', u'\u0f10', u'\u0f11', u'\u0f12', u'\u0f13', u'\u0f14',
    u'\u0f34', u'\u0f35', u'\u0f36', u'\u0f37', u'\u0f38', u'\u0f39', u'\u0f3a',
    u'\u0f3b', u'\u0f3c', u'\u0f3d', u'\u0f84', u'\u0f85', u'\u0fd0', u'\u0fd1', 
    u'\u0fd2', u'\u0fd3', u'\u0fd4')

shad = (u'\u0f06', u'\u0f07', u'\u0f08', u'\u0f0d', u'\u0f0e', u'\u0f0f', u'\u0f10', u'\u0f11', u'\u0f12')

syllables = (u'\u0f00',)
logotype = (u'\u0f15', u'\u0f16')
astro_sign = (u'\u0f17', u'\u0f18', u'\u0f19')
digit = (u'\u0f20', u'\u0f21', u'\u0f22', u'\u0f23', u'\u0f24', u'\u0f25', u'\u0f26',
    u'\u0f27', u'\u0f28', u'\u0f29', u'\u0f2a', u'\u0f2b', u'\u0f2c', u'\u0f2d',
    u'\u0f2e', u'\u0f2f', u'\u0f30', u'\u0f31', u'\u0f32', u'\u0f33')

symbol = (u'\u0fc4', u'\u0fc5', u'\u0fc6', u'\u0fc7', u'\u0fc8', u'\u0fc9',
    u'\u0fca', u'\u0fcb', u'\u0fcc')

norm_roots = {
    u'\u0f41': u'\u0f41', u'\u0f40': u'\u0f40', u'\u0f43': u'\u0f43', 
    u'\u0f42': u'\u0f42', u'\u0f45': u'\u0f45', u'\u0f44': u'\u0f44', 
    u'\u0f47': u'\u0f47', u'\u0f46': u'\u0f46', u'\u0f49': u'\u0f49', 
    u'\u0f4b': u'\u0f4b', u'\u0f4a': u'\u0f4a', u'\u0f4d': u'\u0f4d', 
    u'\u0f4c': u'\u0f4c', u'\u0f4f': u'\u0f4f', u'\u0f4e': u'\u0f4e', 
    u'\u0f51': u'\u0f51', u'\u0f50': u'\u0f50', u'\u0f53': u'\u0f53', 
    u'\u0f52': u'\u0f52', u'\u0f55': u'\u0f55', u'\u0f54': u'\u0f54', 
    u'\u0f57': u'\u0f57', u'\u0f56': u'\u0f56', u'\u0f59': u'\u0f59', 
    u'\u0f58': u'\u0f58', u'\u0f5b': u'\u0f5b', u'\u0f5a': u'\u0f5a', 
    u'\u0f5d': u'\u0f5d', u'\u0f5c': u'\u0f5c', u'\u0f5f': u'\u0f5f', 
    u'\u0f5e': u'\u0f5e', u'\u0f61': u'\u0f61', u'\u0f60': u'\u0f60', 
    u'\u0f63': u'\u0f63', u'\u0f62': u'\u0f62', u'\u0f65': u'\u0f65', 
    u'\u0f64': u'\u0f64', u'\u0f67': u'\u0f67', u'\u0f66': u'\u0f66', 
    u'\u0f69': u'\u0f69', u'\u0f68': u'\u0f68', u'\u0f6b': u'\u0f6b', 
    u'\u0f6a': u'\u0f6a', u'\u0f6c': u'\u0f6c',
    u'\u0f91': u'\u0f41', u'\u0f90': u'\u0f40', u'\u0f93': u'\u0f43', 
    u'\u0f92': u'\u0f42', u'\u0f95': u'\u0f45', u'\u0f94': u'\u0f44', 
    u'\u0f97': u'\u0f47', u'\u0f96': u'\u0f46', u'\u0f99': u'\u0f49', 
    u'\u0f9b': u'\u0f4b', u'\u0f9a': u'\u0f4a', u'\u0f9d': u'\u0f4d', 
    u'\u0f9c': u'\u0f4c', u'\u0f9f': u'\u0f4f', u'\u0f9e': u'\u0f4e', 
    u'\u0fa1': u'\u0f51', u'\u0fa0': u'\u0f50', u'\u0fa3': u'\u0f53', 
    u'\u0fa2': u'\u0f52',     u'\u0fa5': u'\u0f55', u'\u0fa4': u'\u0f54', 
    u'\u0fa7': u'\u0f57', u'\u0fa6': u'\u0f56',     u'\u0fa9': u'\u0f59', 
    u'\u0fa8': u'\u0f58', u'\u0fab': u'\u0f5b', u'\u0faa': u'\u0f5a',
    u'\u0fad': u'\u0f5d', u'\u0fac': u'\u0f5c', u'\u0faf': u'\u0f5f', 
    u'\u0fae': u'\u0f5e',     u'\u0fb1': u'\u0f61', u'\u0fb0': u'\u0f60', 
    u'\u0fb3': u'\u0f63', u'\u0fb2': u'\u0f62', u'\u0fb5': u'\u0f65', 
    u'\u0fb4': u'\u0f64', u'\u0fb7': u'\u0f67', u'\u0fb6': u'\u0f66', 
    u'\u0fb9': u'\u0f69', u'\u0fb8': u'\u0f68', u'\u0fbb': u'\u0f6b', 
    u'\u0fba': u'\u0f6a',     u'\u0fbc': u'\u0f6c'
    }

word_parts = letters + subjoined_letters + f_vowels
non_letters = signs + syllables + marks + logotype + astro_sign + digit + symbol + (u"\n", u"\r", u" ", u"\t", u"\u00A0")
non_letters2 = signs + syllables + marks + logotype + astro_sign + digit + symbol

word_parts_set = frozenset(word_parts)
non_letters_set = frozenset(non_letters)


lexical_map = {"root_only":root_only, "subjoined":subjoined, 
                        "subjoined_cons":subjoined_consonants, 
                        "vowel":f_vowels, "non_letter":non_letters}

