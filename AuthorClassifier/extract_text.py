import re
import pandas as pd

# Define the texts
hemingway = """
In the sentry box that faced toward them up the road, the sentry was sitting holding his rifle, the bayonet fixed, between his knees.
He was smoking a cigarette and he wore a knitted cap and blanket style cape.
At fifty yards, you could not see anything about his face.
Robert put up his field glasses, shading the lenses carefully with his cupped hands even though there was now no sun to make a glint, and there was the rail of the bridge as clear as though you could reach out and touch it and there was the face of the sentry so clear he could see the sunken cheeks, the ash on the cigarette and the greasy shine of the bayonet.
It was a peasant's face, the cheeks hollow under the high cheekbones, the beard stubbled, the eyes shaded by the heavy brows, big hands holding the rifle, heavy boots showing beneath the folds of the blanket cape.
There was a worn, blackened leather wine bottle on the wall of the sentry box, there were some newspapers and there was no telephone.
There could, of course, be a telephone on the side he could not see; but there were no wires running from the box that were visible.
A telephone line ran along the road and its wires were carried over the bridge.
There was a charcoal brazier outside the box, made from an old petrol tin with the top cut off and holes punched in it, which rested on two stones; but he held no fire.
There were some fire-blackened empty tins in the ashes under it.
Robert Jordan handed the glasses to Anselmo who lay flat beside him.
The old man grinned and shook his head.
He tapped his skull beside his eye with one finger.
"Ya io veo," he said in Spanish.
"I have seen him," speaking from the front of his mouth with almost no movement of his lips in the way that is quieter than any whisper.
He looked at the sentry as Robert Jordan smiled at him and, pointing with one finger, drew the other across his throat.
Robert Jordan nodded but he did not smile.
The sentry box at the far end of the bridge faced away from them and down the road and they could not see into it.
The road, which was broad and oiled and well constructed, made a turn to the left at the far end of the bridge and then swung out of sight around a curve to the right.
At this point it was enlarged from the old road to its present width by cutting into the solid bastion of the rock on the far side of the gorge; and its left or western edge, looking down from the pass and the bridge, was marked and protected by a line of upright cut blocks of stone where its edge fell sheer away to the gorge.
The gorge was almost a canyon here, where the brook, that the bridge was flung over, merged with the main stream of the pass.
He was watching the sentry again with his glasses.
The sentry rubbed his cigarette out on the plank wall of the box, then took a leather tobacco pouch from his  pocket, opened the paper of the dead cigarette and emptied the remnant of used tobacco into the pouch.
The sentry stood up, leaned his rifle against the wall of the box and stretched, picked up his rifle, slung it over his shoulder and walked out onto the bridge.
Anselmo flattened on the ground and Robert Jordan slipped his glasses into his shirt pocket and put his head well behind the pine tree.
The sentry was standing, his back toward them, at the far end of the bridge.
From the gorge came the noise of the stream in the boulders.
Then through this noise came another noise, a steady, racketing drone and they saw the sentry looking up, his knitted cap slanted back, and turning their heads and looking up they saw, high in the evening sky, three monoplanes in V formation, showing minute and silvery at that height where there still was sun, passing unbelievably quickly across the sky, their motors now throbbing steadily.
Robert Cohn was once middleweight boxing champion of Princeton.
Do not think that I am very much impressed by that as a boxing title, but it meant a lot to Cohn.
He cared nothing for boxing, in fact he disliked it, but he learned it painfully and thoroughly to counteract the feeling of inferiority and shyness he had felt on being treated as a Jew at Princeton.
There was a certain inner comfort in knowing he could knock down anybody who was snooty to him, although, being very shy and a thoroughly nice boy, he never fought except in the gym.
He was Spider Kelly’s star pupil.
Spider Kelly taught all his young gentlemen to box like featherweights, no matter whether they weighed one hundred and five or two hundred and five pounds.
But it seemed to fit Cohn.
He was really very fast.
He was so good that Spider promptly overmatched him and got his nose permanently flattened.
This increased Cohn’s distaste for boxing, but it gave him a certain satisfaction of some strange sort, and it certainly improved his nose.
In his last year at Princeton he readtoo much and took to wearing spectacles.
I never met any one of his class who remembered him.
They did not even remember that he was middleweight boxing champion.
I mistrust all frank and simple people, especially when their stories hold together, and I always had a suspicion that perhaps Robert Cohn had never been middleweight boxing champion, and that perhaps a horse had stepped on his face, or that maybe his mother had been frightened or seen something, or that he had, maybe, bumped into something as a young child, but I finally had somebody verify the story from Spider Kelly.
Spider Kelly not only remembered Cohn. He had often wondered what hadbecome of him.
Robert Cohn was a member, through his father, of one of the richest Jewish families in New York, and through his mother of one of the oldest.
At the military school where he prepped for Princeton, and played a very good end on the football team, no one had made him race-conscious.
No one had ever made him feel he was a Jew, and hence any different from anybody else, until he went to Princeton.
He was a nice boy, a friendly boy, and very shy, and it made him bitter.
He took it out in boxing, and he came out of Princeton with painful self-consciousness and the flattened nose, and was married by the first girl who was nice to him.
He was married five years, had three children, lost most of the fifty thousand dollars his father left him, the balance of the estate having gone to his mother, hardened into a rather unattractive mould under domestic unhappiness with a rich wife; and just when he had made up his mind to leave his wife she left him and went off with a miniature-painter.
As he had been thinking for months about leaving his wife and had not done it because it would be too cruel to deprive her of himself, her departure was a very healthful shock.
"""

lawrence = """
Connie helped him as much as she could. At first she was thrilled. He talked everything over with her monotonously, insistently, persistently, and she had to respond with all her might. It was as if her whole soul and body and sex had to rouse up and pass into theme stories of his. This thrilled her and absorbed her.
Of physical life they lived very little. She had to superintend the house. But the housekeeper had served Sir Geoffrey for many years, and the dried-up, elderly, superlatively correct female you could hardly call her a parlour-maid, or even a woman...who waited at table, had been in the house for forty years. Even the very housemaids were no longer young. It was awful! What could you do with such a place, but leave it alone! All these endless rooms that nobody used, all the Midlands routine, the mechanical cleanliness and the mechanical order! Clifford had insisted on a new cook, an experienced woman who had served him in his rooms in London. For the rest the place seemed run by mechanical anarchy. Everything went on in pretty good order, strict cleanliness, and strict punctuality; even pretty strict honesty. And yet, to Connie, it was a methodical anarchy. No warmth of feeling united it organically. The house seemed as dreary as a disused street.
What could she do but leave it alone? So she left it alone. Miss Chatterley came sometimes, with her aristocratic thin face, and triumphed, finding nothing altered. She would never forgive Connie for ousting her from her union in consciousness with her brother. It was she, Emma, who should be bringing forth the stories, these books, with him; the Chatterley stories, something new in the world, that THEY, the Chatterleys, had put there. There was no other standard. There was no organic connexion with the thought and expression that had gone before. Only something new in the world: the Chatterley books, entirely personal.
Connie's father, where he paid a flying visit to Wragby, and in private to his daughter: As for Clifford's writing, it's smart, but there's NOTHING IN IT. It won't last! Connie looked at the burly Scottish knight who had done himself well all his life, and her eyes, her big, still-wondering blue eyes became vague. Nothing in it! What did he mean by nothing in it? If the critics praised it, and Clifford's name was almost famous, and it even brought in money...what did her father mean by saying there was nothing in Clifford's writing? What else could there be?
The Brangwens went home to Beldover, the wedding-party gathered at Shortlands, the Criches' home. It was a long, low old house, a sort of manor farm, that spread along the top of a slope just beyond the narrow little lake of Willey Water. Shortlands looked across a sloping meadow that might be a park, because of the large, solitary trees that stood here and there, across the water of the narrow lake, at the wooded hill that successfully hid the colliery valley beyond, but did not quite hide the rising smoke. Nevertheless, the scene was rural and picturesque, very peaceful, and the house had a charm of its own.
It was crowded now with the family and the wedding guests. The father, who was not well, withdrew to rest. Gerald was host. He stood in the homely entrance hall, friendly and easy, attending to the men. He seemed to take pleasure in his social functions, he smiled, and was abundant in hospitality.
The women wandered about in a little confusion, chased hither and thither by the three married daughters of the house. All the while there could be heard the characteristic, imperious voice of one Crich woman or another calling 'Helen, come here a minute,' 'Marjory, I want you--here.' 'Oh, I say, Mrs Witham--.' There was a great rustling of skirts, swift glimpses of smartly-dressed women, a child danced through the hall and back again, a maidservant came and went hurriedly.
Meanwhile the men stood in calm little groups, chatting, smoking, pretending to pay no heed to the rustling animation of the women's world. But they could not really talk, because of the glassy ravel of women's excited, cold laughter and running voices. They waited, uneasy, suspended, rather bored. But Gerald remained as if genial and happy, unaware that he was waiting or unoccupied, knowing himself the very pivot of the occasion.
They felt the rush of the sap in spring, they knew the wave which cannot halt, but every year throws forward the seed to begetting, and, falling back, leaves the young-born on the earth. They knew the intercourse between heaven and earth, sunshine drawn into the breast and bowels, the rain sucked up in the daytime, nakedness that comes under the wind in autumn, showing the birds' nests no longer worth hiding.
"""

wharton = """
Before my own time there was up I had learned to know what that meant.
Yet I had come in the degenerate day of trolley, bicycle and rural
delivery, when communication was easy between the scattered mountain
villages, and the bigger towns in the valleys, such as Bettsbridge and
Shadd's Falls, had libraries, theatres and Y. M. C. A. halls to which
the youth of the hills could descend for recreation. But when winter
shut down on Starkfield and the village lay under a sheet of snow
perpetually renewed from the pale skies, I began to see what life
there--or rather its negation--must have been in Ethan Frome's young
manhood.

I had been sent up by my employers on a job connected with the big
power-house at Corbury Junction, and a long-drawn carpenters' strike
had so delayed the work that I found myself anchored at Starkfield--the
nearest habitable spot--for the best part of the winter. I chafed at
first, and then, under the hypnotising effect of routine, gradually
began to find a grim satisfaction in the life. During the early part of
my stay I had been struck by the contrast between the vitality of
the climate and the deadness of the community. Day by day, after the
December snows were over, a blazing blue sky poured down torrents
of light and air on the white landscape, which gave them back in an
intenser glitter. One would have supposed that such an atmosphere must
quicken the emotions as well as the blood; but it seemed to produce
no change except that of retarding still more the sluggish pulse of
Starkfield. When I had been there a little longer, and had seen this
phase of crystal clearness followed by long stretches of sunless cold;
when the storms of February had pitched their white tents about the
devoted village and the wild cavalry of March winds had charged down to
their support; I began to understand why Starkfield emerged from its
six months' siege like a starved garrison capitulating without quarter.
Twenty years earlier the means of resistance must have been far fewer,
and the enemy in command of almost all the lines of access between the
beleaguered villages; and, considering these things, I felt the sinister
force of Harmon's phrase: “Most of the smart ones get away.” But if that
were the case, how could any combination of obstacles have hindered the
flight of a man like Ethan Frome?

During my stay at Starkfield I lodged with a middle-aged widow
colloquially known as Mrs. Ned Hale. Mrs. Hale's father had been the
village lawyer of the previous generation, and “lawyer Varnum's house,”
 where my landlady still lived with her mother, was the most considerable
mansion in the village. It stood at one end of the main street, its
classic portico and small-paned windows looking down a flagged path
between Norway spruces to the slim white steeple of the Congregational
church. It was clear that the Varnum fortunes were at the ebb, but the
two women did what they could to preserve a decent dignity; and Mrs.
Hale, in particular, had a certain wan refinement not out of keeping
with her pale old-fashioned house.

In the “best parlour,” with its black horse-hair and mahogany weakly
illuminated by a gurgling Carcel lamp, I listened every evening to
another and more delicately shaded version of the Starkfield chronicle.
It was not that Mrs. Ned Hale felt, or affected, any social superiority
to the people about her; it was only that the accident of a finer
sensibility and a little more education had put just enough distance
between herself and her neighbours to enable her to judge them with
detachment. She was not unwilling to exercise this faculty, and I had
great hopes of getting from her the missing facts of Ethan Frome's
story, or rather such a key to his character as should co-ordinate the
facts I knew. Her mind was a store-house of innocuous anecdote and any
question about her acquaintances brought forth a volume of detail; but
on the subject of Ethan Frome I found her unexpectedly reticent. There
was no hint of disapproval in her reserve; I merely felt in her an
insurmountable reluctance to speak of him or his affairs, a low “Yes, I
knew them both... it was awful...” seeming to be the utmost concession
that her distress could make to my curiosity.

So marked was the change in her manner, such depths of sad initiation
did it imply, that, with some doubts as to my delicacy, I put the case
anew to my village oracle, Harmon Gow; but got for my pains only an
uncomprehending grunt.

“Ruth Varnum was always as nervous as a rat; and, come to think of it,
she was the first one to see 'em after they was picked up. It happened
right below lawyer Varnum's, down at the bend of the Corbury road, just
round about the time that Ruth got engaged to Ned Hale. The young folks
was all friends, and I guess she just can't bear to talk about it. She's
had troubles enough of her own.”

All the dwellers in Starkfield, as in more notable communities, had had
troubles enough of their own to make them comparatively indifferent to
those of their neighbours; and though all conceded that Ethan Frome's
had been beyond the common measure, no one gave me an explanation of the
look in his face which, as I persisted in thinking, neither poverty
nor physical suffering could have put there. Nevertheless, I might have
contented myself with the story pieced together from these hints had
it not been for the provocation of Mrs. Hale's silence, and--a little
later--for the accident of personal contact with the man.

He did not in the least wish the future Mrs. Newland Archer to be a
simpleton.  He meant her (thanks to his enlightening companionship) to
develop a social tact and readiness of wit enabling her to hold her own
with the most popular married women of the "younger set," in which it
was the recognised custom to attract masculine homage while playfully
discouraging it.  If he had probed to the bottom of his vanity (as he
sometimes nearly did) he would have found there the wish that his wife
should be as worldly-wise and as eager to please as the married lady
whose charms had held his fancy through two mildly agitated years;
without, of course, any hint of the frailty which had so nearly marred
that unhappy being's life, and had disarranged his own plans for a
whole winter.

How this miracle of fire and ice was to be created, and to sustain
itself in a harsh world, he had never taken the time to think out; but
he was content to hold his view without analysing it, since he knew it
was that of all the carefully-brushed, white-waistcoated,
button-hole-flowered gentlemen who succeeded each other in the club
box, exchanged friendly greetings with him, and turned their
opera-glasses critically on the circle of ladies who were the product
of the system.  In matters intellectual and artistic Newland Archer
felt himself distinctly the superior of these chosen specimens of old
New York gentility; he had probably read more, thought more, and even
seen a good deal more of the world, than any other man of the number.
Singly they betrayed their inferiority; but grouped together they
represented "New York," and the habit of masculine solidarity made him
accept their doctrine on all the issues called moral.  He instinctively
felt that in this respect it would be troublesome--and also rather bad
form--to strike out for himself.

But they had; they undoubtedly had; for the low-toned comments behind
him left no doubt in Archer's mind that the young woman was May
Welland's cousin, the cousin always referred to in the family as "poor
Ellen Olenska."  Archer knew that she had suddenly arrived from Europe
a day or two previously; he had even heard from Miss Welland (not
disapprovingly) that she had been to see poor Ellen, who was staying
with old Mrs. Mingott.  Archer entirely approved of family solidarity,
and one of the qualities he most admired in the Mingotts was their
resolute championship of the few black sheep that their blameless stock
had produced.  There was nothing mean or ungenerous in the young man's
heart, and he was glad that his future wife should not be restrained by
false prudery from being kind (in private) to her unhappy cousin; but
to receive Countess Olenska in the family circle was a different thing
from producing her in public, at the Opera of all places, and in the
very box with the young girl whose engagement to him, Newland Archer,
was to be announced within a few weeks.  No, he felt as old Sillerton
Jackson felt; he did not think the Mingotts would have tried it on!
"""

Fitzgerald = """
He didn’t say any more, but we’ve always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that. In consequence, I’m inclined to reserve all judgments, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores. The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, because I was privy to the secret grief’s of wild, unknown men. Most of the confidences were unsought—frequently I have feigned sleep, preoccupation, or a hostile levity when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon; for the intimate revelations of young men, or at least the terms in which they express them, are usually plagiaristic and marred by obvious suppressions. Reserving judgments is a matter of infinite hope. I am still a little afraid of missing something if I forget that, as my father snobbishly suggested, and I snobbishly repeat, a sense of the fundamental decencies is parceled out unequally at birth.

And, after boasting this way of my tolerance, I come to the admission that it has a limit. Conduct may be founded on the hard rock or the wet marshes, but after a certain point I don’t care what it’s founded on. When I came back from the East last autumn I felt that I wanted the world to be in uniform and at a sort of moral attention forever; I wanted no more riotous excursions with privileged glimpses into the human heart. Only Gatsby, the man who gives his name to this book, was exempt from my reaction—Gatsby, who represented everything for which I have an unaffected scorn. If personality is an unbroken series of successful gestures, then there was something gorgeous about him; some heightened sensitivity to the promises of life, as if he were related to one of those intricate machines that register earthquakes ten thousand miles away. This responsiveness had nothing to do with that flabby impressionability which is dignified under the name of the “creative temperament”—it was an extraordinary gift for hope, a romantic readiness such as I have never found in any other person and which it is not likely I shall ever find again. No—Gatsby turned out all right at the end; it is what preyed on Gatsby, what foul dust floated in the wake of his dreams that temporarily closed out my interest in the abortive sorrows and short-winded elations of men.

We drove for a long time over a bright level countryside, just a
road and a tree and a shack and a tree, and then suddenly along a
winding twist of woodland. I could feel even in the darkness that
the trees of the woodland were green-— that it was all different from
the dusty olive-tint of California. Somewhere we passed a negro
driving three cows ahead of him, and they mooed as he scatted them
to the side of the road. They were real cows, with warm, fresh, silky
flanks, and the negro grew gradually real out of the darkness with
his big brown eyes staring at us close to the car, as Wylie gave him

“Let’s all ask for our money back,” he suggested. His dark eyes
took me in, and I wondered what they would look like if he fell in
love. They were kind, aloof and, though they often reasoned with
you gently, somewhat superior. It was no fault of theirs if they saw
He darted in and out of the role of “one of the boys” with
dexterity—but on the whole I should say he wasn’t one of them.
But he knew how to shut up, how to draw into the background,
how to listen. From where he stood (and though he was not a tall
man, it always seemed high up) he watched the multitudinous practicalities of his world like a proud young shepherd to whom night
and day had never mattered. He was born sleepless, without a talent
so much.

We sat in unembarrassed silence—I had known him since he
became Father’s partner a dozen years ago, when I was seven and
Stahr was twenty-two. Wylie was across the aisle and I didn’t know
whether or not to introduce them, but Stahr kept turning his ring
so abstractedly that he made me feel young and invisible, and I
didn’t dare. I never dared look quite away from him or quite at him,
unless I had something important to say—and I knew he affected
many other people in the same manner.

   Indeed, of all the region only the beach stirred with activity.
Three British nannies sat knitting the slow pattern of
Victorian England, the pattern of the forties, the sixties, and
the eighties, into sweaters and socks, to the tune of gossip as
formalized as incantation; closer to the sea a dozen persons
kept house under striped umbrellas, while their dozen children
pursued unintimidated fish through the shallows or lay naked
and glistening with cocoanut oil out in the sun.
   As Rosemary came onto the beach a boy of twelve ran past
her and dashed into the sea with exultant cries. Feeling the
impactive scrutiny of strange faces, she took off her bathrobe
and followed. She floated face down for a few yards and find-
ing it shallow staggered to her feet and plodded forward,
dragging slim legs like weights against the resistance of the
water. When it was about breast high, she glanced back toward
shore: a bald man in a monocle and a pair of tights, his tufted
chest thrown out, his brash navel sucked in, was regarding
her attentively. As Rosemary returned the gaze the man dis-
lodged the monocle, which went into hiding amid the facetious
whiskers of his chest, and poured himself a glass of
something from a bottle in his hand.
   Rosemary laid her face on the water and swam a choppy
little four-beat crawl out to the raft. The water reached up
for her, pulled her down tenderly out of the heat, seeped in
her hair and ran into the comers of her body. She turned round
and round in it, embracing it, wallowing in it. Reaching the raft
she was out of breath, but a tanned woman with very white
teeth looked down at her, and Rosemary, suddenly conscious
of the raw whiteness of her own body, turned on her back
and drifted toward shore. The hairy man holding the bottle
spoke to her as she came out.

Amory Blaine inherited from his mother every trait, except the stray inexpressible few, that made him worthwhile. His father, an ineffectual, inarticulate man with a taste for Byron and a habit of drowsing over the Encyclopedia Britannica, grew wealthy at thirty through the death of two elder brothers, successful Chicago brokers, and in the first flush of feeling that the world was his, went to Bar Harbor and met Beatrice O’Hara. In consequence, Stephen Blaine handed down to posterity his height of just under six feet and his tendency to waver at crucial moments, these two abstractions appearing in his son Amory. For many years he hovered in the background of his family’s life, an unassertive figure with a face half-obliterated by lifeless, silky hair, continually occupied in “taking care” of his wife, continually harassed by the idea that he didn’t and couldn’t understand her.
But Beatrice Blaine! There was a woman! Early pictures taken on her father’s estate at Lake Geneva, Wisconsin, or in Rome at the Sacred Heart Convent—an educational extravagance that in her youth was only for the daughters of the exceptionally wealthy—showed the exquisite delicacy of her features, the consummate art and simplicity of her clothes. A brilliant education she had—her youth passed in renaissance glory, she was versed in the latest gossip of the Older Roman Families; known by name as a fabulously wealthy American girl to Cardinal Vitori and Queen Margherita and more subtle celebrities that one must have had some culture even to have heard of. She learned in England to prefer whiskey and soda to wine, and her small talk was broadened in two senses during a winter in Vienna. All in all Beatrice O’Hara absorbed the sort of education that will be quite impossible ever again; a tutelage measured by the number of things and people one could be contemptuous of and charming about; a culture rich in all arts and traditions, barren of all ideas, in the last of those days when the great gardener clipped the inferior roses to produce one perfect bud.
The sixty acres of the estate were dotted with old and new summer houses and many fountains and white benches that came suddenly into sight from foliage-hung hiding-places; there was a great and constantly increasing family of white cats that prowled the many flowerbeds and were silhouetted suddenly at night against the darkening trees. It was on one of the shadowy paths that Beatrice at last captured Amory, after Mr. Blaine had, as usual, retired for the evening to his private library. After reproving him for avoiding her, she took him for a long tête-à-tête in the moonlight. He could not reconcile himself to her beauty, that was mother to his own, the exquisite neck and shoulders, the grace of a fortunate woman of thirty.
"""

Orwell = """
By degrees they were issuing the recruits with uniforms, and because
this was Spain everything was issued piecemeal, so that it was never
quite certain who had received what, and various of the things we most
needed, such as belts and cartridge-boxes, were not issued till the last
moment, when the train was actually waiting to take us to the front. I
have spoken of the militia 'uniform', which probably gives a wrong
impression. It was not exactly a uniform. Perhaps a 'multiform' would be
the proper name for it. Everyone's clothes followed the same general
plan, but they were never quite the same in any two cases. Practically
everyone in the army wore corduroy knee-breeches, but there the
uniformity ended. Some wore puttees, others corduroy gaiters, others
leather leggings or high boots. Everyone wore a zipper jacket, but some
of the jackets were of leather, others of wool and of every conceivable
colour. The kinds of cap were about as numerous as their wearers. It was
usual to adorn the front of your cap with a party badge, and in addition
nearly every man wore a red or red and black handkerchief round his
throat. A militia column at that time was an extraordinary-looking
rabble. But the clothes had to be issued as this or that factory rushed
them out, and they were not bad clothes considering the circumstances.
The shirts and socks were wretched cotton things, however, quite useless
against cold. I hate to think of what the militiamen must have gone
through in the earlier months before anything was organized. I remember
coming upon a newspaper of only about two months earlier in which one of
the P.O.U.M. leaders, after a visit to the front, said that he would try
to see to it that 'every militiaman had a blanket'. A phrase to make you
shudder if you have ever slept in a trench.

At one end of the big barn, on a sort of raised platform, Major was
already ensconced on his bed of straw, under a lantern which hung from a
beam. He was twelve years old and had lately grown rather stout, but he
was still a majestic-looking pig, with a wise and benevolent appearance in
spite of the fact that his tushes had never been cut. Before long the
other animals began to arrive and make themselves comfortable after their
different fashions. First came the three dogs, Bluebell, Jessie, and
Pincher, and then the pigs, who settled down in the straw immediately in
front of the platform. The hens perched themselves on the window-sills,
the pigeons fluttered up to the rafters, the sheep and cows lay down
behind the pigs and began to chew the cud. The two cart-horses, Boxer and
Clover, came in together, walking very slowly and setting down their vast
hairy hoofs with great care lest there should be some small animal
concealed in the straw. Clover was a stout motherly mare approaching
middle life, who had never quite got her figure back after her fourth foal.
Boxer was an enormous beast, nearly eighteen hands high, and as strong as
any two ordinary horses put together. A white stripe down his nose gave
him a somewhat stupid appearance, and in fact he was not of first-rate
intelligence, but he was universally respected for his steadiness of
character and tremendous powers of work. After the horses came Muriel,
the white goat, and Benjamin, the donkey. Benjamin was the oldest animal
on the farm, and the worst tempered. He seldom talked, and when he did, it
was usually to make some cynical remark--for instance, he would say that
God had given him a tail to keep the flies off, but that he would sooner
have had no tail and no flies. Alone among the animals on the farm he
never laughed. If asked why, he would say that he saw nothing to laugh at.
Nevertheless, without openly admitting it, he was devoted to Boxer; the
two of them usually spent their Sundays together in the small paddock
beyond the orchard, grazing side by side and never speaking.

The shop was a narrow, cold sort of room. On the outside of the window
a few white letters, relics of ancient chocolate advertisements, were
scattered like stars. Inside there was a slab upon which lay the great
white folds of tripe, and the grey flocculent stuff known as 'black
tripe', and the ghostly translucent feet of pigs, ready boiled. It was
the ordinary 'tripe and pea' shop, and not much else was stocked except
bread, cigarettes, and tinned stuff. 'Teas' were advertised in the
window, but if a customer demanded a cup of tea he was usually put off
with excuses. Mr Brooker, though out of work for two years, was a miner
by trade, but he and his wife had been keeping shops of various kinds as
a side-line all their lives. At one time they had had a pub, but they
had lost their licence for allowing gambling on the premises. I doubt
whether any of their businesses had ever paid; they were the kind of
people who run a business chiefly in order to have something to grumble
about. Mr Brooker was a dark, small-boned, sour, Irish-looking man, and
astonishingly dirty. I don't think I ever once saw his hands clean. As
Mrs Brooker was now an invalid he prepared most of the food, and like
all people with permanently dirty hands he had a peculiarly intimate,
lingering manner of handling things. If he gave you a slice of
bread-and-butter there was always a black thumb-print on it. Even in the
early morning when he descended into the mysterious den behind Mrs
Brooker's sofa and fished out the tripe, his hands were already black. I
heard dreadful stories from the other lodgers about the place where the
tripe was kept. Blackbeetles were said to swarm there. I do not know how
often fresh consignments of tripe were ordered, but it was at long
intervals, for Mrs Brooker used to date events by it. 'Let me see now,
I've had in three lots of froze (frozen tripe) since that happened,'
etc. We lodgers were never given tripe to eat. At the time I imagined
that this was because tripe was too expensive; I have since thought that
it was merely because we knew too much about it. The Brookers never ate
tripe themselves, I noticed.

For some reason the telescreen in the living-room was in an unusual
position. Instead of being placed, as was normal, in the end wall, where
it could command the whole room, it was in the longer wall, opposite the
window. To one side of it there was a shallow alcove in which Winston
was now sitting, and which, when the flats were built, had probably been
intended to hold bookshelves. By sitting in the alcove, and keeping well
back, Winston was able to remain outside the range of the telescreen, so
far as sight went. He could be heard, of course, but so long as he stayed
in his present position he could not be seen. It was partly the unusual
geography of the room that had suggested to him the thing that he was now
about to do.

But it had also been suggested by the book that he had just taken out of
the drawer. It was a peculiarly beautiful book. Its smooth creamy paper,
a little yellowed by age, was of a kind that had not been manufactured for
at least forty years past. He could guess, however, that the book was much
older than that. He had seen it lying in the window of a frowsy little
junk-shop in a slummy quarter of the town (just what quarter he did not
now remember) and had been stricken immediately by an overwhelming desire
to possess it. Party members were supposed not to go into ordinary shops
('dealing on the free market', it was called), but the rule was not
strictly kept, because there were various things, such as shoelaces and
razor blades, which it was impossible to get hold of in any other way. He
had given a quick glance up and down the street and then had slipped inside
and bought the book for two dollars fifty. At the time he was not conscious
of wanting it for any particular purpose. He had carried it guiltily home
in his briefcase. Even with nothing written in it, it was a compromising
possession.
"""

Kerouac = """
Then came spring, the great time of traveling, and everybody in the scattered gang was getting ready to take one trip or another. I was busily at work on my novel and when I carne to the halfway mark, after a trip down South with my aunt to visit my brother Rocco, I got ready to travel West for the very first time.
Dean had already left. Carlo and I saw him off at the 34th Street Greyhound station. Upstairs they had a place where you could make pictures for a quarter. Carlo took off his glasses and looked sinister. Dean made a profile shot and looked coyly around. I took a straight picture that made me look like a thirty-year-old Italian who’d kill anybody who said anything against his mother. This picture Carlo and Dean neatly cut down the middle with a razor and saved a half each in their
wallets. Dean was wearing a real Western business suit for his big trip back to Denver; he’d finished his first fling in New York. I say fling, but he only worked like a dog in parking lots. The most fantastic parking-lot attendant in the world, he can back a car forty miles an hour into a tight squeeze and stop at the wall, jump out, race among fenders, leap into another car, circle it fifty miles an hour in a narrow space, back swiftly into tight spot, hump, snap the car with the emergency so that you see it bounce as he flies out; then clear to the ticket shack, sprinting like a track star, hand a ticket, leap into a newly arrived car before the owner’s half out, leap literally under him as he steps out, start the car with the door flapping, and roar off to the next available spot, arc, pop in, brake, out, run; working like that without pause eight hours a night, evening rush hours and after-theater rush hours, in greasy wino pants with a frayed fur-lined jacket and beat shoes that flap. Now he’d bought a new
suit to go back in; blue with pencil stripes, vest and all - eleven dollars on Third Avenue, with a watch and watch chain, and a portable typewriter with which he was going to start writing in a Denver rooming house as soon as he got a job there. We had a farewell meal of franks and beans in a Seventh Avenue Riker’s, and then Dean got on the bus that said Chicago and roared off into the night. There went our wrangler. I promised myself to go the same way when spring really bloomed and opened up the land.
In the month of July 1947, having saved about fifty dollars from old veteran benefits, I was ready to go to the West Coast. My friend Remi Boncœur had written me a letter from San Francisco, saying I should come and ship out with him on an around-the-world liner. He swore he could get me into the engine room. I wrote back and said I’d be satisfied with any old freighter so long as I could take a few long Pacific trips and come back with enough money to support myself in my aunt’s house while I finished my book. He said he had a shack in Mill City and I would have all the time in the world to write there while we went through the rigmarole of getting the ship. He was living with a girl called Lee Ann; he said she was a marvelous cook and everything would jump. Remi was an old prep-school friend, a Frenchman brought up in Paris and a really mad guy - I didn’t know how mad at this time. So he expected me to arrive in ten days. My aunt was all in accord with my trip to the West; she said it would do me good, I’d been working so hard all winter and staying in too much; she even didn’t complain when I told her I’d have to hitchhike some. All she wanted was for me to come back in one piece. So, leaving my big half-manuscript sitting on top of my desk, and folding back my comfortable home sheets for the last time one morning, I left with my canvas bag in which a few fundamental things were packed and took off for the Pacific Ocean with the fifty dollars in my pocket.

I jumped over the side and ran across Highway 101 to the store, and bought, besides wine,
a little bread and candy. I ran back to my freight train which had another fifteen minutes to
wait in the now warm sunny scene. But it was late afternoon and bound to get cold soon. The
little bum was sitting cross-legged at his end before a pitiful repast of one can of sardines. I
took pity on him and went over and said, "How about a little wine to warm you up? Maybe
you'd like some bread and cheese with your sardines."
"Sure thing." He spoke from far away inside a little meek voice-box afraid or unwilling to
assert himself. I'd bought the cheese three days ago in Mexico City before the long cheap bus
trip across Zacatecas and Durango and Chihuahua two thousand long miles to the border at El
Paso. He ate the cheese and bread and drank the wine with gusto and gratitude. I was pleased.
I reminded myself of the line in the Diamond Sutra that says, "Practice charity without
holding in mind any conceptions about charity, for charity after all is just a word." I was very
devout in those days and was practicing my religious devotions almost to perfection. Since
then I've become a little hypocritical about my lip-service and a little tired and cynical.
Because now I am grown so old and neutral. But then I really believed in the reality of
charity and kindness and humility and zeal and neutral tranquillity and wisdom and ecstasy,
and I believed that I was an old time bhikku in modern clothes wandering the world (usually
the immense triangular arc of New York to Mexico City to San Francisco) in order to turn the
wheel of the True Meaning, or Dharma, and gain merit for myself as a future Buddha
(Awakener) and as a future Hero in Paradise. I had not met Japhy Ryder yet, I was about to
the next week, or heard anything about "Dharma Bums" although at this time I was a perfect
Dharma Bum myself and considered myself a religious wanderer. The little bum in the
gondola solidified all my beliefs by warming up to the wine and talking and finally whipping
out a tiny slip of paper which contained a prayer by Saint Teresa announcing that after her
death she will return to the earth by showering it with roses from heaven, forever, for all
living creatures.
And I look around the dismal cell, there’s my hopeful rucksack all neatly packed with everything necessary to live in the woods, even unto the minutest first aid kit and diet details and even a neat little sewing kit cleverly reinforced by my good mother (like extra safety pins, buttons, special sewing needles, little aluminum scissors)… The hopeful medal of St Christopher even which she’d sewn on the flap… The survival kit all in there down to the last little survival sweater and handkerchief and tennis sneakers (for hiking) — But the rucksack sits hopefully in a strewn mess of bottles all empty, empty poor boys of white port, butts, junk, horror… “One fast move or I’m gone, “ I realize, gone the way of the last three years of drunken hopelessness which is a physical and spiritual and metaphysical hopelessness you cant learn in school no matter how many books on existentialism or pessimism you read, or how many jugs of vision producing Ayahuasca you drink, or Mescaline take, or Peyote goop up with — That feeling when you wake up with the delirium tremens with the fear of eerie death dripping from your ears like those special heavy cobwebs spiders weave in the hot countries, the feeling of being a bent back mudman monster groaning underground in hot steaming mud pulling a long hot burden nowhere, the feeling of standing ankledeep in hot boiled pork blood, ugh, of being up to your waist in a giant pan of greasy brown dishwater not a trace of suds left in it… The face of yourself you see in the mirror with its expression of unbearable anguish so haggard and awful with sorrow you cant even cry for a thing so ugly, so lost, no connection whatever with early perfection and therefore nothing to connect with tears or anything: it’s like William Seward Burroughs” “Stranger” suddenly appearing in your place in the mirror — Enough! “One fast move or I’m gone” so I jump up, do my headstand first to pump blood back into the hairy brain, take a shower in the hall, new T-shirt and socks and underwear, pack vigorously, hoist the rucksack and run out throwing the key on the desk and hit the cold street and walk fast to the nearest little grocery store to buy two days of food, stick it in the rucksack, hike thru lost alleys of Russian sorrow where bums sit head on knees in foggy doorways in the goopy eerie city night I’ve got to escape or die, and into the bus station In a half hour into a bus seat, the bus says “Monterey” and off we go down the clean neon hiway and I sleep all the way, waking up amazed and well again smelling sea air the bus driver shaking me “End of the line, Monterey. “ — And by God it is Monterey, I stand sleepy in the 2 A. M. seeing vague little fishing masts across the street from the bus driveway. Now all I’ve got to do to complete my escape is get 14 miles down the coast to the Raton Canyon bridge and hike in.
"""

Austen = """
In respect of her art generally, Mr. Goldwin Smith has truly observed
that “metaphor has been exhausted in depicting the perfection of it,
combined with the narrowness of her field;” and he has justly added that
we need not go beyond her own comparison to the art of a miniature
painter. To make this latter observation quite exact we must not use the
term miniature in its restricted sense, and must think rather of Memling
at one end of the history of painting and Meissonier at the other, than
of Cosway or any of his kind. And I am not so certain that I should
myself use the word “narrow” in connection with her. If her world is a
microcosm, the cosmic quality of it is at least as eminent as the
littleness. She does not touch what she did not feel herself called to
paint; I am not so sure that she could not have painted what she did not
feel herself called to touch. It is at least remarkable that in two very
short periods of writing--one of about three years, and another of not
much more than five--she executed six capital works, and has not left a
single failure. It is possible that the romantic paste in her
composition was defective: we must always remember that hardly
anybody born in her decade--that of the eighteenth-century
seventies--independently exhibited the full romantic quality. Even Scott
required hill and mountain and ballad, even Coleridge metaphysics and
German to enable them to chip the classical shell. Miss Austen was an
English girl, brought up in a country retirement, at the time when
ladies went back into the house if there was a white frost which might
pierce their kid shoes, when a sudden cold was the subject of the
gravest fears, when their studies, their ways, their conduct were
subject to all those fantastic limits and restrictions against which
Mary Wollstonecraft protested with better general sense than particular
taste or judgment. Miss Austen, too, drew back when the white frost
touched her shoes; but I think she would have made a pretty good journey
even in a black one.

No sooner was his father’s funeral over, than Mrs. John Dashwood,
without sending any notice of her intention to her mother-in-law,
arrived with her child and their attendants. No one could dispute her
right to come; the house was her husband’s from the moment of his
father’s decease; but the indelicacy of her conduct was so much the
greater, and to a woman in Mrs. Dashwood’s situation, with only common
feelings, must have been highly unpleasing;—but in _her_ mind there was
a sense of honor so keen, a generosity so romantic, that any offence of
the kind, by whomsoever given or received, was to her a source of
immovable disgust. Mrs. John Dashwood had never been a favourite with
any of her husband’s family; but she had had no opportunity, till the
present, of showing them with how little attention to the comfort of
other people she could act when occasion required it.

So acutely did Mrs. Dashwood feel this ungracious behaviour, and so
earnestly did she despise her daughter-in-law for it, that, on the
arrival of the latter, she would have quitted the house for ever, had
not the entreaty of her eldest girl induced her first to reflect on the
propriety of going, and her own tender love for all her three children
determined her afterwards to stay, and for their sakes avoid a breach
with their brother.

Elinor, this eldest daughter, whose advice was so effectual, possessed
a strength of understanding, and coolness of judgment, which qualified
her, though only nineteen, to be the counsellor of her mother, and
enabled her frequently to counteract, to the advantage of them all,
that eagerness of mind in Mrs. Dashwood which must generally have led
to imprudence. She had an excellent heart;—her disposition was
affectionate, and her feelings were strong; but she knew how to govern
them: it was a knowledge which her mother had yet to learn; and which
one of her sisters had resolved never to be taught.

The event had every promise of happiness for her friend. Mr. Weston was
a man of unexceptionable character, easy fortune, suitable age, and
pleasant manners; and there was some satisfaction in considering with
what self-denying, generous friendship she had always wished and
promoted the match; but it was a black morning’s work for her. The want
of Miss Taylor would be felt every hour of every day. She recalled her
past kindness—the kindness, the affection of sixteen years—how she had
taught and how she had played with her from five years old—how she had
devoted all her powers to attach and amuse her in health—and how nursed
her through the various illnesses of childhood. A large debt of
gratitude was owing here; but the intercourse of the last seven years,
the equal footing and perfect unreserve which had soon followed
Isabella’s marriage, on their being left to each other, was yet a
dearer, tenderer recollection. She had been a friend and companion such
as few possessed: intelligent, well-informed, useful, gentle, knowing
all the ways of the family, interested in all its concerns, and
peculiarly interested in herself, in every pleasure, every scheme of
hers—one to whom she could speak every thought as it arose, and who had
such an affection for her as could never find fault.

How was she to bear the change? It was true that her friend was going
only half a mile from them; but Emma was aware that great must be the
difference between a Mrs. Weston, only half a mile from them, and a
Miss Taylor in the house; and with all her advantages, natural and
domestic, she was now in great danger of suffering from intellectual
solitude. She dearly loved her father, but he was no companion for her.
He could not meet her in conversation, rational or playful.
"""

Faulkner = """
When the shadow of the sash appeared on the curtains it was between seven and eight o'clock and then I was in time again, hearing the watch. It was Grandfather's and when Father gave it to me he said I give you the mausoleum of all hope and desire; it's rather excruciatingly apt that you will use it to gain the reductio ad absurdum of all human experience which can fit your individual needs no better than it fitted his or his father's. I give it to you not that you may remember time, but that you might forget it now and then for a moment and not spend all your breath trying to conquer it. Because no battle is ever won he said. They are not even fought. The field only reveals to man his own folly and despair, and victory is an illusion of philosophers and fools.

It was propped against the collar box and I lay listening to it. Hearing it, that is. I don't suppose anybody ever deliberately listens to a watch or a clock. You don't have to. You can be oblivious to the sound for a long while, then in a second of ticking it can create in the mind unbroken the long diminishing parade of time you didn't hear. Like Father said down the long and lonely light-rays you might see Jesus walking, like. And the good Saint Francis that said Little Sister Death, that never had a sister.

Through the wall I heard Shreve's bed-springs and then his slippers on the floor hishing. I got up and went to the dresser and slid my hand along it and touched the watch and turned it face-down and went back to bed. But the shadow of the sash was still there and I had learned to tell almost to the minute, so I'd have to turn my back to it, feeling the eyes animals used to have in the back of their heads when it was on top, itching. It's always the idle habits you acquire which you will regret. Father said that. That Christ was not crucified: he was worn away by a minute clicking of little wheels. That had no sister.

And so as soon as I knew I couldn't see it, I began to wonder what time it was. Father said that constant speculation regarding the position of mechanical hands on an arbitrary dial which is a symptom of mind-function. Excrement Father said like sweating. And I saying All right. Wonder. Go on and wonder.

There were perhaps five families there when Lena arrived. There was a track and a station, and once a day a mixed train fled shrieking through it. The train could be stopped with a red flag, but by ordinary it appeared out of the devastated hills with apparitionlike suddenness and wailing like a banshee, athwart and past that little less-than-village like a forgotten bead from a broken string. The brother was twenty years her senior. She hardly remembered him at all when she came to live with him. He lived in a four room and unpainted house with his labor- and child-ridden wife. For almost half of every year the sister-in-law was either lying in or recovering. During this time Lena did all the housework and took care of the other children. Later she, told herself, ‘I reckon that’s why I got one so quick myself.’

She slept in a lean-to room at the back of the house. It had a window which she learned to open and close again in the dark without making a sound, even though there also slept in the lean-to room at first her oldest nephew and then the two oldest and then the three. She had lived there eight years before she opened the window for the first time. She had not opened it a dozen times hardly before she discovered that she should not have opened it at all. She said to herself, ‘That’s just my luck.’

The sister-in-law told the brother. Then he remarked her changing shape, which he should have noticed some time before. He was a hard man. Softness and gentleness and youth (he was just forty) and almost everything else except a kind of stubborn and despairing fortitude and the bleak heritage of his bloodpride had been sweated out of him. He called her whore. He accused the right man (young bachelors, or sawdust Casanovas anyway, were even fewer in number than families) but she would not admit it, though the man had departed six months ago. She just repeated stubbornly, “He’s going to send for me. He said he would send for me”; unshakable, sheeplike, having drawn upon that reserve of patient and steadfast fidelity upon which the Lucas Burches depend and trust, even though they do not intend to be present when the need for it arises. Two weeks later she climbed again through the window.

And at night it is better still. I used to lie on the pallet in the hall, waiting until I could hear them all asleep, so I could get up and go back to the bucket. It would be black, the shelf black, the still surface of the water a round orifice in nothingness, where before I stirred it awake with the dipper I could see maybe a star or two in the bucket, and maybe in the dipper a star or two before I drank. After that I was bigger, older. Then I would wait until they all went to sleep so I could lie with my shirt-tail up, hearing them asleep, feeling myself without touching myself, feeling the cool silence blowing upon my parts and wondering if Cash was yonder in the darkness doing it too, had been doing it perhaps for the last two years before I could have wanted to or could have.

Pa's feet are badly splayed, his toes cramped and bent and warped, with no toenail at all on his little toes, from working so hard in the wet in homemade shoes when he was a boy. Beside his chair his brogans sit. They look as though they had been hacked with a blunt axe out of pig-iron. Vernon has been to town. I have never seen him go to town in overalls. His wife, they say. She taught school too, once.

I fling the dipper dregs to the ground and wipe my mouth on my sleeve. It is going to rain before morning. Maybe before dark. "Down to the barn," I say. "Harnessing the team."

Down there fooling with that horse. He will go on through the barn, into the pasture. The horse will not be in sight: he is up there among the pine seedlings, in the cool. Jewel whistles, once and shrill. The horse snorts, then Jewel sees him, glinting for a gaudy instant among the blue shadows. Jewel whistles again; the horse comes dropping down the slope, stiff-legged, his ears cocking and flicking, his mis-matched eyes rolling, and fetches up twenty feet away, broadside on, watching Jewel over his shoulder in an attitude kittenish and alert.

It's because she wants it told, he thought, so that people whom she will never see and whose names she will never hear and who have never heard her name nor seen her face will read it and know at last why God let us lose the war: that only through the blood of our men and the tears of our women could He slay this demon and efface his name and lineage from the earth. Then almost immediately he decided that neither was this the reason why she had sent the note, and sending it, why to him, since if she had merely wanted it told, written, and even printed, she would not have needed to call in anybody—a woman who even in his (Quentin's) father's youth had already established herself as the town's and the county's poetess laureate by issuing to the stern and meager subscription list of the county newspaper poems, ode, eulogy, and epitaph, out of some bitter and implacable reserve of undefeat.

It would be three hours yet before he would learn why she had sent for him because part of it, the first part of it, Quentin already knew.

It was a part of his twenty years' heritage of breathing the same ai? and hearing his father talk about the man Sutpen; a part of the town's—Jefferson's—eighty years' heritage of the same air which the man himself had breathed between this September afternoon in 1909 and that Sunday morning in June in 1833 when he first rode into town out of no discernible past and acquired his land no one knew how and built his house, his mansion, apparently out of nothing and married Ellen Coldfield and begot his two children—the son who widowed the daughter who had not yet been a bride—and so accomplished his allotted course to its violent (Miss Coldfield at least would have said, just) end.

Quentin had grown up with that; the mere names were interchangeable and almost myriad. His childhood was full of them; his very body was an empty hall echoing with sonorous defeated names; he was not a being, an entity, he was a commonwealth.
"""

# Split the text into distinct sentences using regular expressions
def create_dataframe(text, prefix):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    df = pd.DataFrame({'Sentence': sentences})
    df['Source'] = prefix
    return df

# Create DataFrames for each text
hemingway_df = create_dataframe(hemingway, 'Hemingway')
lawrence_df = create_dataframe(lawrence, 'Lawrence')
wharton_df = create_dataframe(wharton.replace('\n', ' '), 'Wharton')
Fitzgerald_df = create_dataframe(Fitzgerald.replace('\n', ' '), 'Fitzgerald')
Orwell_df = create_dataframe(Orwell.replace('\n', ' '), 'Orwell')
Kerouac_df = create_dataframe(Kerouac.replace('\n', ' '), 'Kerouac')
Austen_df = create_dataframe(Austen.replace('\n', ' '), 'Austen')
Faulkner_df = create_dataframe(Faulkner.replace('\n', ' '), 'Faulkner')

#combined_df.to_csv("sentences.csv")

# Concatenate the DataFrames into a single DataFrame
combined_df = pd.concat([hemingway_df, lawrence_df,
                         wharton_df, Fitzgerald_df,
                         Austen_df, Faulkner_df,
                         Orwell_df, Kerouac_df], ignore_index = True)

# Print the combined DataFrame
print(combined_df)

combined_df.to_csv("sentences.csv")