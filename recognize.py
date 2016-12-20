#! /usr/bin/python
# encoding: utf-8
'''Primary routines that manage OCR recognition'''
from PIL import Image
from bisect import bisect, bisect_left
import cPickle as pickle
from classify import load_cls
import codecs
from config_manager import Config
from cv2 import drawContours
import cv2 as cv
import datetime
from fast_utils import fadd_padding, ftrim
from feature_extraction import normalize_and_extract_features
from line_breaker import LineCut, LineCluster
import logging
import numpy as np
import os
from page_elements2 import PageElements as PE2
from random import choice
from root_based_finder import is_non_std, word_parts
from segment import Segmenter, combine_many_boxes
import shelve
import signal
import simplejson as json
from sklearn.externals import joblib
import sys
from termset import syllables

from tparser import parse_syllables
from utils import local_file
from viterbi_cython import viterbi_cython
# from viterbi_search import viterbi_search, word_bigram
import warnings

cls = load_cls('logistic-cls')

## Ignore warnings. THis is mostlu in response to incessant sklearn
## warnings about passing in 1d arrays
warnings.filterwarnings("ignore")
print 'ignoring all warnings'
###

rbfcls = load_cls('rbf-cls')
predict_log_proba = cls.predict_log_proba
predict_proba = cls.predict_proba

# Trained characters are labeled by number. Open the shelve that contains
# the mappings between the Unicode character and its number label.
allchars = shelve.open(local_file('allchars_dict2'))
char_to_dig = allchars['allchars']
dig_to_char = allchars['label_chars']
allchars.close()

## Uncomment the line below when enabling viterbi_hidden_tsek
gram3 = pickle.load(open(local_file('3gram_stack_dict.pkl'),'rb'))

word_parts = set(word_parts)

PCA_TRANS = False

trs_prob = np.load(open(local_file('stack_bigram_mat.npz')))
trs_prob = trs_prob[trs_prob.files[0]]

cdmap = pickle.load(open(local_file('extended_char_dig.pkl')))

# HMM data structures
trans_p = np.load(open(local_file('stack_bigram_logprob32.npz')))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = np.load(open(local_file('stack_start_logprob32.npz')))
start_p = start_p[start_p.files[0]]

start_p_nonlog = np.exp(start_p)

## Uncomment below for syllable bigram
syllable_bigram = pickle.load(open(local_file('syllable_bigram.pkl'), 'rb')) #THIS ONE

def get_trans_prob(stack1, stack2):
    try:
        return trs_prob[cdmap[stack1], cdmap[stack2]]
    except KeyError:
        print 'Warning: Transition matrix char-dig map has not been updated with new chars'
        return .25


#############################################
### Post-processing functions ###
#############################################

def viterbi(states, start_p, trans_p, emit_prob):
    '''A basic viterbi decoder implementation
    
    states: a vector or list of states 0 to n
    start_p: a matrix or vector of start probabilities
    trans_p: a matrix of transition probabilities
    emit_prob: an nxT matrix of per-class output probabilities
        where n is the number of states and t is the number
        of transitions
    '''
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] * emit_prob[0][y]
        path[y] = [y]
        
    # Run Viterbi for t > 0
    for t in range(1,len(emit_prob)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_prob[t][y], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[len(emit_prob) - 1][y], y) for y in states])
    return ''.join(dig_to_char[s] for s in path[state])

def viterbi_hidden_tsek(states, start_p, trans_p, emit_prob):
    '''Given a series of recognized characters, infer
likely positions of missing punctuation
    
    Parameters
    --------
    states: the possible classes that can be assigned to (integer codes of stacks)
    start_p: pre-computed starting probabilities of Tibetan syllables
    trans_p: pre-computed transition probabilities between Tibetan stacks
    emit_prob: matrix of per-class probability for t steps
    
    Returns:
    List of possible string candidates with tsek inserted
    '''
    V = [{}]
    path = {}
    tsek_dig = char_to_dig[u'་']
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_prob[0][y]
        path[y] = [y]
    num_obs = len(emit_prob)
    # Run Viterbi for t > 0
    for t in range(1,num_obs*2-1):
        V.append({})
        newpath = {}

        if t % 2 == 1:                
            prob_states = []
            for y0 in states:
                im_path = path.get(y0)
                if not im_path:
                    continue
                if len(im_path) > 1:
                    run_without_tsek = 0
                    for i in im_path[::-1]:
                        if i != tsek_dig:
                            run_without_tsek += 1
                        else:
                            break
                    pr3 = gram3.get(path[y0][-2], {}).get(path[y0][-1],{}).get(tsek_dig,.5)*(1+run_without_tsek*2)
                else:
                    pr3 = .75
                
                try:
                    prob_states.append((V[t-1][y0]*trans_p[y0][tsek_dig]*pr3, y0))
                except:
                    print '-'*20
                    print trans_p[y0]
                    print V[t-1]
                    print '-'*20
                    raise
            prob, state = max(prob_states)
            V[t][tsek_dig] = prob
            newpath[tsek_dig] = path[state] + [tsek_dig]
            path.update(newpath)
        else:
            srted = np.argsort(emit_prob[t/2])

            for y in srted[-50:]:
                #### normal
#                prob, state = max([(V[t-2][y0]*trans_p[y0][y]*emit_prob[t/2][y], y0) for y0 in states])
                ####
                
                #### Experimental
                prob_states = []
                for y0 in states:
                    im_path = path.get(y0,[])[-4:] # immediate n-2 in path
                    t_m2 = V[t-2].get(y0)
                    if not im_path or not t_m2:
                        continue
                    
                    prob_states.append((V[t-2][y0]*trans_p[y0][y]*emit_prob[t/2][y], y0))
                if not prob_states:
                    continue
                prob, state = max(prob_states)
                
                tsek_prob, tsek_dig = (V[t-1][tsek_dig]*trans_p[tsek_dig][y]*emit_prob[t/2][y], tsek_dig)
                
                if tsek_prob > prob:
                    prob = tsek_prob
                    state = tsek_dig
                
                V[t][y] = prob
                newpath[y] = path[state] + [y]
                
            path = newpath
        if not V[t].keys():
            raise ValueError
        (prob, state) = max([(V[t][y], y) for y in V[t].keys()])
    (prob, state) = max([(V[len(V)-1][y], y) for y in V[len(V)-1].keys()])
        
    str_perms = _get_tsek_permutations(''.join(dig_to_char[s] for s in path[state]))
    return str_perms

def _get_tsek_permutations(tsr):
    tsek_count = tsr.count(u'་')
    syls = parse_syllables(tsr, omit_tsek=False)

    all_candidates = []
    if tsek_count > 8:
        print 'too many permutations'
        return [tsr]
    elif tsek_count == 0:
        print 'no tsek'
        return [tsr]
    else:
        ops = [['0','1'] for i in range(tsek_count)]
        allops = iter(_enumrate_full_paths(ops))
        for op in allops:
            nstr = []
            op = list(op[::-1])
            for i in syls:
                if i == u'་' :
                    cur_op = op.pop()
                    if cur_op == '0':
                        continue
                    else:
                        nstr.append(i)
                else:
                    nstr.append(i)
            
            nstr = ''.join(nstr)
            new_parse = parse_syllables(nstr)
            for p in new_parse:
                if is_non_std(p) and p not in syllables:
                    print nstr, 'rejected'
                    break
            else:
                print nstr, 'accepted'
                all_candidates.append(nstr)
    if len(all_candidates) == 0:
        all_candidates = [tsr]
    return all_candidates
        
def hmm_recognize(segmentation):
    '''Only used in speical case where doing tsek-insertion post-process
    
    Parameters:
    __________
    segmentioatn: a segmentation object
    
    
    Returns
    _______
    A tuple (prob, string) corresponding the probability of a
    segmented and recognized string, and its probability
    
    '''
    nstates = trs_prob.shape[0]
    states = range(start_p.shape[0])
    
    obs = []
    bxs = []
    for num, line in enumerate(segmentation.vectors):
        line_boxes = segmentation.new_boxes[num]
        for obn, ob in enumerate(line):
            if not isinstance(ob, unicode):
                obs.append(ob.flatten())
                bxs.append(line_boxes[obn])
            else:
                print ob,
                print 'hmm omitting unicode part'
    if bxs:
        outbox = list(combine_many_boxes(bxs))
    else:
        print 'RETURNED NONE'
        return (0, '')

    emit_p = cls.predict_proba(obs)
    results = []
    syllable = []
    for em in emit_p:
        char = dig_to_char[np.argmax(em)]
        if char in (u'་', u'།'):
            if syllable:
                prob, res = viterbi_hidden_tsek(states, start_p, trs_prob, syllable)
                results.append(res)
                results.append(char)
                syllable = []
        else:
            syllable.append(em)
    if syllable:
        prob, hmm_out = viterbi_hidden_tsek(states, start_p, trs_prob, syllable)
        results.append(hmm_out)
    else:
        prob = 0
        hmm_out = ''
    
    results = ''.join(results)
    print results, '<---RESULT'
    return (prob, results)

def _enumrate_full_paths(tree):
    if len(tree) == 1:
        return tree[0]
    combs = []
    frow = tree[-1]
    srow = tree[-2]
    
    for s in srow:
        for f in frow:
            combs.append(s+f)
    tree.pop()
    tree.pop()
    tree.append(combs)
    return _enumrate_full_paths(tree)

def bigram_prob(syl_list):
    return np.prod([syllable_bigram.get(syl_list[i], {}).get(syl_list[i+1], 1e-5) \
                    for i in range(len(syl_list) -1 )])

def max_syllable_bigram(choices):
    best_prob = 0.0
    best_s = ''
    for s in choices:
        print s, 'is a choice'
        if not isinstance(s, list):
            s = parse_syllables(s)
        prob = bigram_prob(s)
        if prob > best_prob:
            best_prob = prob
            best_s = s
    best_s = u'་'.join(best_s)
    return best_prob, best_s

def hmm_recognize_bigram(segmentation):
    states = range(start_p.shape[0])    
    obs = []
    bxs = []
    for num, line in enumerate(segmentation.vectors):
        line_boxes = segmentation.new_boxes[num]
        for obn, ob in enumerate(line):
            if hasattr(ob, 'flatten'):
                obs.append(ob.flatten())
                bxs.append(line_boxes[obn])
            else:
                print ob,
                print 'hmm omitting unicode part'

    if not obs:
        return (0, '')
    
    emit_p = cls.predict_proba(obs)

    results = []
    syllable = []
    for em in emit_p:
        char = dig_to_char[np.argmax(em)]
        if char in (u'་', u'།'):
            if syllable:

                res = viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable)

                results.append(res)
                results.append(char)
                syllable = []
        else:
            syllable.append(em)
    if syllable:
        hmm_out = viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable)
        
        results.append(hmm_out)
    else:
        prob = 0
        hmm_out = ''

    all_paths = _enumrate_full_paths(results)
    prob, results = max_syllable_bigram(all_paths)
        
    print results, 'RESULTS'
    return (prob, results)

#############################################
### Recognizers
#############################################

def recognize_chars(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: Unicode string containing recognized text'''
    
    results = []

    tsek_mean = segmentation.final_box_info.tsek_mean
    width_dists = {}
    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print 'no vectors...'
            continue
        
        tmp_result = []
        new_boxes = segmentation.new_boxes[l]
        
        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr

        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []

        for s in small_chars[::-1]: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
#        for s in small_chars: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
            cnt = segmentation.line_info.shapes.contours[s]
            bx = segmentation.line_info.shapes.get_boxes()[s]
            bx = list(bx)
            x,y,w,h = bx
            char_arr = np.ones((h,w), dtype=np.uint8)
            offset = (-x, -y)
            drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
            feature_vect = normalize_and_extract_features(char_arr)
            prd = classify(feature_vect)

            insertion_pos = bisect(left_edges, x)

            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                # insertion is at or near end of line and needs more left 
                # neighbors to compensate for there being less chars to define the baseline
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12

            if tsek_insert_method == 'baseline':
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                ####
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    if j[1] + j[3] > bottom:
                        bottom = j[1] + j[3]
                local_span = bottom - top

                if prd == u'་' and local_span > 0:

                    left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                    right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                    local_baseline_left = top + left_sum.argmin()
                    if mid != right:
                        local_baseline_right = top + right_sum.argmin()
                    else:
                        local_baseline_right = local_baseline_left
                    
                    if ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                    (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3])): #or 
#                    (entire_local_baseline >= bx[1] and entire_local_baseline <= bx[1] + bx[3])):
                        ### Account for fact that the placement of a tsek could be 
                        # before or after its indicated insertion pos
                        ### experimental.. only need with certain fonts e.g. "book 6"
                        ## in samples
                        if insertion_pos <= len(new_boxes):
    #                        cur_box_in_pos = new_boxes[insertion_pos]
                            prev_box = new_boxes[insertion_pos-1]
    #                        left_cur = cur_box_in_pos[0]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1

                        vectors.insert(insertion_pos, prd)
                        new_boxes.insert(insertion_pos, bx)
                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])

                elif bx[1] >= top -.25*local_span and bx[1] + bx[3] <= bottom + local_span*.25:
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    left_edges.insert(insertion_pos, bx[0])
            
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                left_edges.insert(insertion_pos, bx[0])
        
        tsek_mean = np.mean(tsek_widths)
        
        for em in emph_markers:
            marker = dig_to_char[segmentation.line_info.shapes.cached_pred_prob[em][0]]
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1]
            bx = segmentation.line_info.shapes.get_boxes()[em]
            bx = list(bx)
            x,y,w,h = bx
            insertion_pos = bisect(left_edges, x)
            bx.append(marker_prob)
            bx.append(marker)
            vectors.insert(insertion_pos, marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])
#        tsek_std = np.std(tsek_widths)
        if len(vectors) == 1: i = -1
        
        for i, v in enumerate(vectors[:-1]):
            if new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2]) >= 2*tsek_mean:
                if not isinstance(v, unicode):
                    prd = classify(v, pca_trans=PCA_TRANS, multi=False)
                else:
                    prd = v

                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
                tmp_result.append([-1,-1,-1,-1, u' '])
            else:
                if not isinstance(v, unicode):
                    prd = classify(v, pca_trans=PCA_TRANS, multi=False)

                    ### Assume that a tsek shouldn't show up at this point
                    ### a more reliable way to do this is to better
#                    if prd == u'་':
#                        prbs = cls.predict_proba(v)[0]
#                        ind_probs = zip(range(len(prbs)), prbs)
#                        ind_probs.sort(key=lambda x: x[1])
#                        prd = dig_to_char[ind_probs[-2][0]]
                else:
                    prd = v
                
                if not width_dists.get(prd):
                    width_dists[prd] = [new_boxes[i][2]]
                else:
                    width_dists[prd].append(new_boxes[i][2])
                
                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
            
        if not isinstance(vectors[-1], unicode):
            prd = classify(vectors[-1], pca_trans=PCA_TRANS, multi=False)
        else:
            prd = vectors[-1]
        new_boxes[-1].append(prd)
        tmp_result.append(new_boxes[-1])
        results.append(tmp_result)

    return results

def recognize_chars_hmm(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page
    '''
    n_states = trans_p.shape[0]
    
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob
#     width_dists = {}
#     times = []
    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print 'no vectors...'
            continue
        
        tmp_result = []
        new_boxes = segmentation.new_boxes[l]
        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr
        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []
        for s in small_chars[::-1]: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
            bx = segmentation.line_info.shapes.get_boxes()[s]
            bx = list(bx)
            x,y,w,h = bx
            try:
                feature_vect = cached_features[s]
                inx, probs = cached_pred_prob[s]
                prob = probs[inx]
                prd = dig_to_char[inx]
#             else:
#                 vect = normalize_and_extract_features(letter)
            except:
                cnt = segmentation.line_info.shapes.contours[s]
                char_arr = np.ones((h,w), dtype=np.uint8)
                offset = (-x, -y)
                drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
                feature_vect = normalize_and_extract_features(char_arr)
#            prd = classify(feature_vect)
                prd, prob = prd_prob(feature_vect)

#            print prd, max(cls.predict_proba(feature_vect)[0])
            insertion_pos = bisect(left_edges, x)
            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12
            if tsek_insert_method == 'baseline':
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                ####
                
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    try:
                        if j[1] + j[3] > bottom:
                            bottom = j[1] + j[3]
                    except IndexError:
                        print new_boxes[lower:upper]
                        print j
                        raise
                local_span = bottom - top

                left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                try:
                    local_baseline_left = top + left_sum.argmin()
                except:
                    local_baseline_left = top 
                    
                if mid != right:
                    local_baseline_right = top + right_sum.argmin()
                else:
                    local_baseline_right = local_baseline_left
                if prd == u'་' and local_span > 0:
                    if ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                    (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3])) or (insertion_pos == len(vectors)): #or 
                        if insertion_pos <= len(new_boxes):
                            prev_box = new_boxes[insertion_pos-1]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1

                        new_boxes.insert(insertion_pos, bx)
                        bx.append(prob)
                        bx.append(prd)
                        vectors.insert(insertion_pos, bx)

                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])

                elif ((bx[1] >= top -.25*local_span and bx[1] + bx[3] <= 
                       bottom + local_span*.25) or 
                      (insertion_pos == len(vectors))) and bx[1] - local_baseline_left < 2*tsek_mean:
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    new_boxes[insertion_pos].append(prob)
                    new_boxes[insertion_pos].append(prd)
                    left_edges.insert(insertion_pos, bx[0])
                    
                else:
                    print 'small contour reject at', l, s, 'local height span', local_span, 'box height', bx[3]
            
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                new_boxes[insertion_pos].append(prob)
                new_boxes[insertion_pos].append(prd)
                left_edges.insert(insertion_pos, bx[0])
        
        for em in emph_markers:
            mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
            marker = dig_to_char[mkinx]
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
            bx = segmentation.line_info.shapes.get_boxes()[em]
            bx = list(bx)
            x,y,w,h = bx
            insertion_pos = bisect(left_edges, x)
            vectors.insert(insertion_pos, marker)
            bx.append(marker_prob)
            bx.append(marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])
        if len(vectors) == 1: i = -1
        
        skip_next_n = 0
        
        ###HMM PHASE

        allstrs = []
        curstr = []
        allinx = []
        curinx = []
        
        for j, v in enumerate(vectors):
            
            islist = isinstance(v, list)
            if isinstance(v, unicode) or islist:
                allstrs.append(curstr)
                allinx.append(curinx)
                curstr = []
                curinx = []
            else:
                curstr.append(v)
                curinx.append(j)
        if curstr:
            allstrs.append(curstr)
            allinx.append(curinx)
        for f, group in enumerate(allstrs):
            if not group: continue
            try:
                probs = predict_log_proba(group)

            except:
                print v,
#                 raise
            LPROB = len(probs)
            if LPROB == 1:
                inx = probs[0].argmax()
                prb = probs[0][inx]
                prds = [inx]
            else:
                probs = probs.astype(np.float32)

                prb, prds = viterbi_cython(LPROB, n_states, start_p, trans_p, probs)
            prb = np.exp(prb)
            inx = allinx[f]
            for vv, c in enumerate(range(len(prds))):
                ind = inx[c]
                cprob = probs[c].max()
                
                #######replace low prob stacks using svm rbf classifier
                ####### warning: this may undo decisions made by hmm classifier
#                 if np.exp(cprob) <= .98:
#  #                     print prds, type(prds)
#                     print 'replacing', dig_to_char[prds[c]], 'with',
#                     prds[c] = rbfcls.predict(group[vv])[0]
# #                    print prds, type(prds)
# #                    print prds[c]
#                     print dig_to_char[prds[c]]
#                     print 
                #######################
                new_boxes[ind].append(np.exp(cprob))
                try:
                    new_boxes[ind].append(dig_to_char[prds[c]])
                except KeyError:
                    new_boxes[ind].append('PROB')
        for ind, b in enumerate(new_boxes):
            tmp_result.append(new_boxes[ind])
            if not len(new_boxes[ind]) == 6:
                print l, ind, new_boxes[ind], '<-----'
            if ind + 1 < len(new_boxes) and  new_boxes[ind+1][0] - (new_boxes[ind][0] + new_boxes[ind][2]) >= 1.5*tsek_mean:
                tmp_result.append([-1,-1,-1,-1, 1.0, u' '])
            
        results.append(tmp_result)
    return results


def recognize_chars_probout(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page'''
    
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob

    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print 'no vectors...'
            continue
        
        tmp_result = []

        new_boxes = segmentation.new_boxes[l]
        scale_w = segmentation.final_box_info.transitions[l]

        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr

        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []
        
        for s in small_chars[::-1]: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
            
            bx = segmentation.line_info.shapes.get_boxes()[s]
            bx = list(bx)
            x,y,w,h = bx
            try:
                feature_vect = cached_features[s]
                inx, probs = cached_pred_prob[s]
                prob = probs[inx]
                prd = dig_to_char[inx]

            except:
                cnt = segmentation.line_info.shapes.contours[s]
                char_arr = np.ones((h,w), dtype=np.uint8)
                offset = (-x, -y)
                drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
                feature_vect = normalize_and_extract_features(char_arr)
                prd, prob = prd_prob(feature_vect)
            
            insertion_pos = bisect(left_edges, x)

            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                # insertion is at or near end of line and needs more left 
                # neighbors to compensate for there being less chars to define the baseline
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12
#            right_items = 5 # bias slightly toward the left. 
            if tsek_insert_method == 'baseline':
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    if j[1] + j[3] > bottom:
                        bottom = j[1] + j[3]
                local_span = bottom - top

                top, bottom, left, right, mid = [int(np.round(ff)) for ff in [top, bottom, left, right, mid]]
                if prd == u'་' and local_span > 0:
                    left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                    right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                    local_baseline_left = top + left_sum.argmin()
                    if mid != right:
                        local_baseline_right = top + right_sum.argmin()
                    else:
                        local_baseline_right = local_baseline_left
                    if ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                    (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3])) or (insertion_pos == len(vectors)): #or 
#                    (entire_local_baseline >= bx[1] and entire_local_baseline <= bx[1] + bx[3])):
                        ### Account for fact that the placement of a tsek could be 
                        # before or after its indicated insertion pos
                        ### experimental.. only need with certain fonts e.g. "book 6"
                        ## in samples
                        if insertion_pos <= len(new_boxes):
                            prev_box = new_boxes[insertion_pos-1]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1
                        
                        vectors.insert(insertion_pos, prd)
                        new_boxes.insert(insertion_pos, bx)
                        new_boxes[insertion_pos].append(prob)
                        new_boxes[insertion_pos].append(prd)
                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])
                elif (bx[1] >= top -.25*local_span and bx[1] + bx[3] <= bottom + local_span*.25) or (insertion_pos == len(vectors)):
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    new_boxes[insertion_pos].append(prob)
                    new_boxes[insertion_pos].append(prd)
                    left_edges.insert(insertion_pos, bx[0])
                    
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                new_boxes[insertion_pos].append(prob)
                new_boxes[insertion_pos].append(prd)
                left_edges.insert(insertion_pos, bx[0])
        
        for em in emph_markers:
            bx = segmentation.line_info.shapes.get_boxes()[em]
            mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
            marker = dig_to_char[mkinx]
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
            
            bx = list(bx)
            x,y,w,h = bx
            bx.append(marker_prob)
            bx.append(marker)
            insertion_pos = bisect(left_edges, x)
            vectors.insert(insertion_pos, marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])

        if len(vectors) == 1: i = -1
        
        skip_next_n = 0
        for i, v in enumerate(vectors[:-1]):

            if skip_next_n:
                skip_next_n -= 1
                continue

            if new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2]) >= 2*tsek_mean:
                if not len(new_boxes[i]) == 6 and not isinstance(v, unicode):
                    prd, prob = prd_prob(v)
                else:
                    if len(new_boxes[i]) == 6:
                        prob, prd = new_boxes[i][4:]
                    else:
                        ## v is unicode stack, likely from segmentation step
                        prd = v
                        prob = .95 # NEED ACTUAL PROB

                new_boxes[i].append(prob)
                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
                tmp_result.append([-1,-1,-1,-1, 1.0, u' '])
            else:
                if hasattr(v, 'dtype'):
                    try:
                        prd, prob = prd_prob(v)
                    except:
                        print v
                    
                    new_boxes[i].append(prob)
                    new_boxes[i].append(prd)
                else:
                    if len(new_boxes[i]) == 6:
                        prob, prd = new_boxes[i][4:]
                    else:
                        prd = v
                
                if len(new_boxes[i]) < 6:
                    try:
                        new_boxes[i].append(prob)
                    except:
                        new_boxes[i].append(1)
                    new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
            
            
        if hasattr(vectors[-1], 'dtype'):
            prd, prob = prd_prob(vectors[-1])
            new_boxes[-1].append(prob)
            new_boxes[-1].append(prd)
        tmp_result.append(new_boxes[-1])
        results.append(tmp_result)
    return results

def viterbi_post_process(img_arr, results):
    '''Go through all results and attempts to correct invalid syllables'''
    final = [[] for i in range(len(results))]
    for i, line in enumerate(results):
        syllable = []
        for j, char in enumerate(line):
            if char[-1] in u'་། ' or not word_parts.intersection(char[-1]) or j == len(line)-1:
                if syllable:
                    syl_str = ''.join(s[-1] for s in syllable)
                    
                    if is_non_std(syl_str) and syl_str not in syllables:
                        print syl_str, 'HAS PROBLEMS. TRYING TO FIX'
                        bx = combine_many_boxes([ch[0:4] for ch in syllable])
                        bx = list(bx)
                        arr = img_arr[bx[1]:bx[1]+bx[3], bx[0]:bx[0]+bx[2]]
                        arr = fadd_padding(arr, 3)
                        try:
                            
                            prob, hmm_res = main(arr, Config(line_break_method='line_cut', page_type='book', postprocess=False, viterbi_postprocess=True, clear_hr=False), page_info={'flname':''})
                        except TypeError:
                            print 'HMM run exited with an error.'
                            prob = 0
                            hmm_res = ''
                        
#                         corrections[syl_str].append(hmm_res) 
                        logging.info(u'VPP Correction: %s\t%s' % (syl_str, hmm_res))
                        if prob == 0 and hmm_res == '':
                            print 'hit problem. using unmodified output'
                            for s in syllable:
                                final[i].append(s)
                        else:
                            bx.append(prob)
                            bx.append(hmm_res)
                            final[i].append(bx)
                    else:
                        for s in syllable:
                            final[i].append(s)
                final[i].append(char)
                syllable = []
            else:
                syllable.append(char)
        if syllable:
            for s in syllable:
                final[i].append(s)

    return final

def main(page_array, conf=Config(viterbi_postprocess=False, line_break_method = None, page_type = None), retries=0,
         text=False, page_info={}):
    '''Main procedure for processing a page from start to finish
    
    Parameters:
    --------------------
    page_array: a 2 dimensional numpy array containing binary pixel data of 
        the image
    
    page_info: dictionary, optional
        A dictionary containing metadata about the page to be recognized.
        Define strings for the keywords "flname" and "volume" if saving
        a serialized copy of the OCR results. 

    retries: Used internally when system attempts to reboot a failed attempt
    
    text: boolean flag. If true, return text rather than char-position data
    
    Returns:
    --------------
    text: str
        Recognized text for entire page
        
    if text=False, return character position and label data as a python dictionary
    '''
    
    print page_info.get('flname','')
    
    confpath = conf.path
    conf = conf.conf
    
    line_break_method = conf['line_break_method']
    page_type = conf['page_type']

    ### Set the line_break method automatically if it hasn't been
    ### specified beforehand
    if not line_break_method and not page_type:
        if page_array.shape[1] > 2*page_array.shape[0]:
            print 'setting page type as pecha'
            line_break_method = 'line_cluster'
            page_type = 'pecha'
        else: 
            print 'setting page type as book'
            line_break_method = 'line_cut'
            page_type = 'book' 
            
    conf['page_type'] = page_type
    conf['line_break_method'] = line_break_method
    detect_o = conf.get('detect_o', False)
    print 'clear hr', conf.get('clear_hr', False)

    results = []
    out = u''
    try:
        ### Get information about the pages
        shapes = PE2(page_array, cls, page_type=page_type, 
                     low_ink=conf['low_ink'], 
                     flpath=page_info.get('flname',''),
                     detect_o=detect_o, 
                     clear_hr =  conf.get('clear_hr', False))
        shapes.conf = conf
        
        ### Separate the lines on a page
        if page_type == 'pecha':
            k_groups = shapes.num_lines
        shapes.viterbi_post = conf['viterbi_postprocess']
        
        if line_break_method == 'line_cut':
            line_info = LineCut(shapes)
            if not line_info: # immediately skip to re-run with LineCluster
                sys.exit()
        elif line_break_method == 'line_cluster':
            line_info = LineCluster(shapes, k=k_groups)
        
        
        ### Perform segmentation of characters
        segmentation = Segmenter(line_info)

        ###Perform recognition
        if not conf['viterbi_postprocess']:
            if conf['recognizer'] == 'probout':
                results = recognize_chars_probout(segmentation)
            elif conf['recognizer'] == 'hmm':
                results = recognize_chars_hmm(segmentation, trans_p, start_p)
            elif conf['recognizer'] == 'kama':
                results = recognize_chars_probout(segmentation)
                results = recognize_chars_kama(results, segmentation)
            if conf['postprocess']:
                results = viterbi_post_process(segmentation.line_info.shapes.img_arr, results)
        else: # Should only be call from *within* a non viterbi run...

            prob, results = hmm_recognize_bigram(segmentation)
            return prob, results
        
        
        ### Construct an output string
        output  = []
        for n, line in enumerate(results):
            for m,k in enumerate(line):
#                 if isinstance(k[-1], int):
#                     print n,m,k
#                     page_array[k[1]:k[1]+k[3], k[0]:k[0]+k[2]] = 0
#                     Image.fromarray(page_array*255).show()
                    
                output.append(k[-1])
            output.append(u'\n')

        out =  ''.join(output)
        print out
    
        if text:
            results = out
        
        return results
    except:
        ### Retry and assume the error was cause by use of the
        ### wrong line_break_method...
        import traceback;traceback.print_exc()
        if not results and not conf['viterbi_postprocess']:
            print 'WARNING', '*'*40
            print page_info['flname'], 'failed to return a result.'
            print 'WARNING', '*'*40
            print
            if line_break_method == 'line_cut' and retries < 1:
                print 'retrying with line_cluster instead of line_cut'
                try:
                    return main(page_array, conf=Config(path=confpath, line_break_method='line_cluster', page_type='pecha'), page_info=page_info, retries = 1, text=text)
                except:
                    logging.info('Exited after failure of second run.')
                    return []
        if not conf['viterbi_postprocess']: 
            if not results:
                logging.info('***** No OCR output for %s *****' % page_info['flname'])
            return results

def run_main(fl, conf=None, text=False):
    '''Helper function to do recognition'''
    if not conf:
#         conf = Config(low_ink=False, segmenter='stochastic', recognizer='hmm', 
#               break_width=2.0, page_type='pecha', line_break_method='line_cluster', 
#               line_cluster_pos='center', postprocess=False, detect_o=False,
#               clear_hr = False)
# 
        conf = Config(segmenter='stochastic', recognizer='hmm', break_width=2.5,  
                      line_break_method='line_cut', postprocess=False,
                      low_ink=False, stop_line_cut=False, clear_hr=True, 
                      detect_o=False)

    return main(np.asarray(Image.open(fl).convert('L'))/255, conf=conf, 
                page_info={'flname':os.path.basename(fl), 'volume': VOL}, 
                text=text)


if __name__ == '__main__':
    fls = ['/Users/zach/random-tibetan-tiff.tif']

    lbmethod = 'line_cluster'
    page_type = 'pecha'
    VOL = 'single_volumes'
    
    def run_main(fl):
        try:
            return main(np.asarray(Image.open(fl).convert('L'))/255, 
                        conf=Config(break_width=2.5, recognizer='hmm', 
                                    segmenter='stochastic', page_type='pecha', 
                                    line_break_method='line_cluster'), 
                        page_info={'flname':fl, 'volume': VOL})
        except:
            return []
    import datetime
    start = datetime.datetime.now()
    print 'starting'
    outfile = codecs.open('/home/zr/latest-ocr-outfile.txt', 'w', 'utf-8')
    
    for fl in fls:
        
        #### line cut
#         ret = main((np.asarray(Image.open(fl).convert('L'))/255), 
#            conf=Config(break_width=2., recognizer='probout', 
#            segmenter='stochastic', line_break_method='line_cut', 
#            postprocess=False, stop_line_cut=False, low_ink=False, clear_hr=True), 
#                    page_info={'flname':fl, 'volume': VOL}, text=True)

        #### line cluster
        ret = main((np.asarray(Image.open(fl).convert('L'))/255), 
                   conf=Config(segmenter='stochastic', recognizer='hmm', 
                               break_width=2.0, page_type='pecha', 
                               line_break_method='line_cluster',
                               line_cluster_pos='center', postprocess=False,
                                detect_o=False, low_ink=False, clear_hr=True), 
                    page_info={'flname':fl, 'volume': VOL}, text=True)
        outfile.write(ret)
        outfile.write('\n\n')

    print datetime.datetime.now() - start, 'time taken'
 
