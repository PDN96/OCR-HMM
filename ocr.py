#!/usr/bin/env python3


# ./ocr.py : Perform optical character recognition, usage:
# ./ocr.py train-image-file.png train-text.txt test-image-file.png


'''
In this problem we are trying to recognize characters in an image. 
The problem has been simplified to a large extend as the following assumptions/simplifications
have been made are made in the question: 
1. The character in the image are one of the following 
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?"'. Hence, there are no unseen characters.
2. Each character fits in a box with is of fixed size. Also, the characters have the same font style and size. 
This would be helpful in classifying the image as the * pizels will be the same in the train and the test image. 
3. We also assume that the text train file provide would properly represent English language. 
4. In English language the transitions for small letters and capital letters is the same. Hence, while calculating the 
initial and transition probabilities we update the probabilities for both small and capital letters.  
5. The train image provided is noise-free. 
O represents the observed imaged.
S represents the actual state (character of the image)
Simple Model: 
For each char i, we have to find Si that maximizes P(Si)p(Oi|Si)
Si can be calculated by counting each character in the train file provided. 
In order to calculate the emission probability a naive bayes classifier is used. 
In the classifier we compare each pixel of the test image with the train image and count the matches. 
Finally the emission probabilities are calculated by using the formula: 
(1 - noise)**count *  noise**(total pixels - count)) 
The images can be noisy. Hence, in order to account for the noise we use the above formula instead of just calculating 
the probability by using count/total. 
This, noise parameter is tunable. Emperically, we found that 0.2 works pretty good. 
Initial and Transition probability for unknown words and transition: 
We add a very small constant to the numerator and denominator. This ensures that even if there is a unknown 
transition in the test, our model doesn't assume that the transition is impossible. 
Hidden Markov Model: 
We are aiming to predict the most likely sequence of character provided the test images. 
Unlike the simple model, this model takes into account the sequence of the characters. 
In english, some seqeunce of charcters are more likely than others. This model takes this fact into account. 
It aims at finding a seqeunce of state(char) that maximize P(S0, S1,.., Sn| O1, O2, ..., On)
In order to calculate the most likely sequence Viterbi algorithm has been used.
There is a significant improvement in the recognition after using HMM.
Key observations:
1. The simple model was failing to distinguish between 1,i and l. The HMM impoved this. 
2. The naive Bayes fails a times to distinguish between c and d. Also the initial and transitions probabilties 
of c and d are very closes. Hence, at times we get the wrong output. 
3. For certain test like test 8. The model Naive Bayes classifier performs very poorly as the image is very noisy. 
It is even difficult for humans to read.
4. The model was outputing space a lot. As it has a very high probability in the test file. Hence, we have adjusted 
the initial probability of space to a smaller value.
Results:
The model have been trained on the courier-train image provided and the cleaned version of the text file from 
part 1 of the assignment.
test 0
simple: SUPREME COURT OF THE UN1TED STATES
viterbi: SUPREME COURT OF THE UNITED STATES
final answer: SUPREME COURT OF THE UNITED STATES
time taken:30.597432374954224 sec 
test 1
simple:          .                   -                1           .
viterbi: . . . .  . .  .   . . .    ' -  '  .   . '  - 1  ' .      . .   1.  1
final answer: . . . .  . .  .   . . .    ' -  '  .   . '  - 1  ' .      . .   1.  1
time taken:53.8809118270874 sec
test 2
simple: Nos. 14-556. Arguec Apr11 28, 2015 - Dec1cec June 26, 2015
viterbi: Nos. 14-556. Argued April 28, 2015 - Deciced June 26, 2015
final answer: Nos. 14-556. Argued April 28, 2015 - Deciced June 26, 2015
time taken:46.626970529556274 sec
test 3
simple: Together w1th No. 14-562, Tanco et a1. v. Has1am, Governor of
viterbi: Together with No. 14-562, Tanco et al. v. Haslam, Governor of
final answer: Together with No. 14-562, Tanco et al. v. Haslam, Governor of
time taken:45.08685541152954 sec
test 4
simple: .enn-sse , -t  1.. a1so c  cent1cr r1 to th.   me  c rt.
viterbi: .enn-sse., et  1.. a so on cent or r. to th. s me co rt.
final answer: .enn-sse., et  1.. a so on cent or r. to th. s me co rt.
time taken:43.07355785369873 sec
test 5
simple: Opinion of.the Ccurt
viterbi: Opinion of the Court
final answer: Opinion of the Court
time taken:23.32254981994629 sec 
test 6
simple:      m-  . . -  -..   n-r  .n  h- - c  -  c-mon .r  -, m   1 g-
viterbi: A  - m-  f . - pe.. 1 n-r  in th- e ca e  c mon tr t-, m r 1 g
final answer: A  - m-  f . - pe.. 1 n-r  in th- e ca e  c mon tr t-, m r 1 g
time taken:46.0757896900177 sec 
test 7 
simple: emcoc1es a 1ove that may endure even past ceath.
viterbi: embocies a love that may endure even past ceath.
final answer: embocies a love that may endure even past ceath.
time taken:38..076212882995605 sec 
test 8
simple: 1     1   1
viterbi: 1  .  1  ..  '  . .    . -   .   .   . m   .    . .  .  . .  . .
final answer: 1  .  1  ..  '  . .    . -   .   .   . m   .    . .  .  . .  . .
time taken:47.315428256988525 sec
test 9 
simple:     . -  cf m  r1 c .
viterbi: . . 1 -a of m' r. c..
final answer: . . 1 -a of m' r. c..
time taken:24.248504400253296 sec
test 10 
simple: Their p1ea is that they do respect it, respect it so deep1y that
viterbi: Their plea is that they co respect it, respect it so ceeply that
final answer: Their plea is that they co respect it, respect it so ceeply that
time taken:46.86928987503052 sec
test 11 
simple:   -,  -       1   1   . 1i111m-   i .   -m -1 - .
viterbi: t -,  --  t   1 c 1 ' . 1.111m- t i . t -m -1 - .
final answer: t -,  --  t   1 c 1 ' . 1.111m- t i . t -m -1 - .
time taken:38.77275085449219 sec
test 12
simple: The1r hope is not to be ccndemned tc 11ve .n 1one1iness,
viterbi: Their hope is not to be condemned to 11.e in lone.iness,
final answer: Their hope is not to be condemned to 11.e in lone.iness,
time taken:42.24308371543884 sec 
test 13 
simple: exc1uded from one of civi1ization's o1dest institutions.
viterbi: excluded from one of civilization's oldest institutions.
final answer: excluded from one of civilization's oldest institutions.
time taken:42.529940128326416 sec
test 14
simple: They ask for equa1 c1gn1ty 1n the eyes of the 1aw.
viterbi: They ask for equa1 cignity in the eyes of the 1aw.
final answer: They ask for equa1 cignity in the eyes of the 1aw.
time taken:40.14026641845703 sec 
test 15 
simple: The Constitution grants them that right.
viterbi: The Constitution grants them that right.
final answer: The Constitution grants them that right.
time taken:33.62787389755249 sec
test 16 
simple: 1 -   c. .-n.  .   - .      . -  e 1  .cr . . ..x h   r  1       v-rs  .
viterbi: . e " c. m n. '.  h- Co. t c. ' ce 1  . r t . ..  h . r  1. 1  r v rs c.
final answer: . e " c. m n. '.  h- Co. t c. ' ce 1  . r t . ..  h . r  1. 1  r v rs c.
time taken:50.28631639480591 sec
test 17
simple: 1t is so ordered.
viterbi: 1t is so ordered.
final answer: 1t is so ordered.
time taken:22.413583040237427 sec
test 18 
simple:        .  .,   1  -  c   -   .      i        . . 1    1
viterbi:  F N D.. "., c 11. r c . - ' . . n  i . - .. . . 1  . 1.
final answer:  F N D.. "., c 11. r c . - ' . . n  i . - .. . . 1  . 1.
time taken:43.15381073951721 sec
test 19
simple: C1N D RC, DRE.L . SO1OM Y  ,  nc K C N,   ..  cin c.
viterbi: CIN D RG, BREYE . S TOM Y D, anc K C N,  "..  cin c.
final answer: CIN D RG, BREYE . S TOM Y D, anc K C N,  "..  cin c.
time taken:40.254761934280396 sec
'''
from PIL import Image, ImageDraw, ImageFont
import sys
import copy
import time


CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

# this function calculated takes a mode input. It will return the most likely work if mode is simple 
# else it returns the emission probability.
def naive_bayes(test_img, char_, mode):
    count_dict = {}
    for alpha in train_letters:
        count_dict[alpha] = 0
        for i,r in enumerate(train_letters[alpha]):
            for j,l in enumerate(r):
                if(train_letters[alpha][i][j] == test_img[i][j]):
                    count_dict[alpha]+=1

    noise = 0.2
    c_dict  = {key:(initial_prob[key]*(1 - noise)**value *  noise**(14*25 - value)) for key,value in count_dict.items()}
    if(mode == "simple"):
        return max(c_dict, key=c_dict.get)
    else:
        return c_dict[char_]/sum(c_dict.values())

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# this function calculates the intitial probabilties from the train text file 
def calc_initial_prob(inp):
    global initial_prob
    total = 0
    init_prob = copy.deepcopy(initial_prob)
    for line in txt_inp:
        for char in line:
            total+=1
            if(char.isalpha()):
                init_prob[char.swapcase()]+=1
                total+=1
            init_prob[char]+= 1

    # smoothening to ensure that probability of unknown word is not 0
    initial_prob = {key:((value+0.0000001)/(total+0.0000001)) for key,value in init_prob.items()}

# this function returns the most likely char sequence by using Viterbi algorithm 
def hmm():
    prob = {}
    for i in TRAIN_LETTERS:
        prob[(0,i)] = (initial_prob[i]*naive_bayes(test_letters[0], i, ""), -1)

    for t in range(1, len(test_letters)): 
        for j in TRAIN_LETTERS:
            emission = naive_bayes(test_letters[t], j,"")
            max_value = -1
            max_element = TRAIN_LETTERS[0]
            for i in TRAIN_LETTERS:
                prob_t = prob[(t-1, i)][0]*transition_prob[(i,j)]*emission
                if(prob_t > max_value):
                    max_element = i
                    max_value = prob_t
                    prob[(t,j)] = (max_value, i)
        sum_deno = 0
        for char in TRAIN_LETTERS:
            sum_deno+=prob[(t,char)][0]
        for char in TRAIN_LETTERS:
            prob[(t,char)] = (prob[(t,char)][0]/sum_deno, prob[(t,char)][1])

    final_char = TRAIN_LETTERS[0]
    for t in TRAIN_LETTERS:
        if(prob[(len(test_letters) -1, t)] > prob[(len(test_letters) -1, final_char)] ):
            final_char = t
    char1 = final_char
    list1 = []
    list1.append(char1)
    for t in range(len(test_letters) -1, 0, -1):
        char1 = prob[(t,char1)][1]
        list1.append(char1)

    print()
    list1.reverse()
    return list1
    [print(a,end="") for a in list1]

# this function calculates the transition probability from the train text file
def calc_transition_prob(inp):
    global transition_prob
    tran_prob = copy.deepcopy(transition_prob)
    for line in txt_inp:
        for n,char in enumerate(line):
            if(n>0):
                if((line[n -1 ]).isalpha() and char.isalpha()): 
                    tran_prob[(line[n - 1].swapcase(), char.swapcase())]+=1

                if((line[n -1 ]).isalpha()):
                    tran_prob[(line[n - 1].swapcase(), char)]+=1

                if(char.isalpha()):
                    tran_prob[(line[n - 1], char.swapcase())]+=1

                tran_prob[(line[n - 1], char)]+= 1

    for key,value in tran_prob.items():

        deno_sum = 0
        for b in TRAIN_LETTERS:
            deno_sum+= tran_prob[(key[0],b)]

        # smoothening to ensure that unknown transition does not have 0 probability.
        if (deno_sum) : transition_prob[key] = (value+0.0000001)/(deno_sum+0.0000001)

start_time = time.time()

# main program
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
with open(train_txt_fname) as f:
    txt_inp = [x.strip() for x in f.read().split('\n')]

with open("test-strings.txt") as f:
    test_file = [x.strip() for x in f.read().split('\n')]

file_no = 2
initial_prob = {key:0 for key in TRAIN_LETTERS}
transition_prob = {(i,j):0 for i in TRAIN_LETTERS for j in TRAIN_LETTERS}
calc_initial_prob(txt_inp)
calc_transition_prob(txt_inp)

initial_prob[' '] = 0.00001

print("simple:",end=" ")
for r in test_letters:
    print(naive_bayes(r,"a","simple"),end="")

list1 = hmm()
print("viterbi:",end=" ")
[print(a,end="") for a in list1]
print()
print("final answer:")
[print(a,end="") for a in list1]
