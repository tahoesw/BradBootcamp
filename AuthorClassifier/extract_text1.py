import re
import pandas as pd

# Define the texts
hemingway = """
Before I put on my coat I cut the cloth stars off my sleeves and put them in the inside pocket with my money. My money was wet but was all right. I counted it. There were three thousand and some lire. My clothes felt wet and clammy and I slapped my arms to keep the circulation going.
I had wovenunderwear and I did not think I would catch cold if I kept moving. They had taken my pistol at the road and I put the holster under my coat. I had no cape and it was cold in the rain. I started up the bank of the canal. It was daylight and the country was wet, low and dismal looking. The fields were bare and wet; a long way away I could see a campanile rising out of the plain. I came up onto a road. Ahead I saw some troops coming down the road. I limped along the side of the road and they passed me and paid no attention to me. They were a machine-gun detachment going up toward the river. I went on down the road.
That day I crossed the Venetian plain. It is a low level country and under the rain it is even flatter. Toward the sea there are salt marshes and very few roads. The roads all go along the river mouths to the sea and to cross the country you must go along the paths beside the canals. I was working across the country from the north to the south and had crossed two railway lines and many roads and finally I came out at the end of a path onto a railway line where it ran beside a marsh. It was the main line from Venice to Trieste, with a high solid embankment, a solid roadbed and double track. Down the tracks a way was a flag- station and I could see soldiers on guard. Up the line there was a bridge over a stream that flowed into the marsh. I could see a guard too at the bridge. Crossing the fields to the north I had seen a train pass on this railroad, visible a long way across the flat plain, and I thought a train might come from Portogruaro. I watched the guards and lay down on the embankment so that I could see both ways along the track.
The guard at the bridgewalked a way up the line toward where flay, then turned and went back toward the bridge. I lay, and was hungry, and waited for the train. The one I had seen was so long that the engine moved it very slowly and I was sure I could get aboard it. After I had almost given up hoping for one I saw a train coming. The engine, coming straight on, grew larger slowly. I looked at the guard at the bridge. He was walking on the near side of the bridge but on the other side of the tracks. That would put him out of sight when the train passed. I watched the engine come nearer. It was working hard. I could see there were many cars. I knew there would be guards on the train, and I tried to see where they were, but, keeping out of sight, I could not. The engine was almost to where I was lying. When it came opposite, working and puffing even on the level, and I saw the engineer pass, I stood up and stepped up close to the passing cars. If the guards were watching I was a less suspicious object standing beside the track. Several closed freight-cars passed. Then I saw a low open car of the sort they call gondolas coming, covered with canvas. I stood until it had almost passed, then jumped and caught the rear hand-rods and pulled up. I crawled down between the gondola and the shelter of the high freight- car behind. I did not think any one had seen me. I was holding to the hand-rods and crouching low, my feet on the coupling. We were almost opposite the bridge. I remembered the guard. As we passed him he looked at me. He was a boy and his helmet was too big for him. I stared at him contemptuously and he looked away. He thought I had something to do with the train.
We were past. I saw him still looking uncomfortable, watching the other cars pass and I stooped to see how the canvas was fastened. It had grummets and was laced down at the edge with cord. I took out my knife, cut the cord and put my arm under. There were hard bulges under the canvas that tightened in the rain. I looked up and ahead. There was a guard on the freight-car ahead but he was looking forward. I let go of the hand-rails and ducked under the canvas. My forehead hit something that gave me a violent bump and I felt blood on my face but I crawled on in and lay flat. Then I turned around and fastened down the canvas.
I was in under the canvas with guns. They smelled cleanly of oil and grease. I lay and listened to the rain on the canvas and the clicking of the car over the rails. There was a little light came through and I lay and looked at the guns. They had their canvas jackets on. I thought they must have been sent ahead from the third army. The bump on my forehead was swollen and I stopped the bleeding by lying still and letting it coagulate, then picked away the dried blood except over the cut. It was nothing. I had no handkerchief, but feeling with my fingers I washed away where the dried blood had been, with rainwater that dripped from the canvas, and wiped it clean with the sleeve of my coat. I did not want to look conspicuous. I knew I would have to get out before they got to Mestre because they would be taking care of these guns. They had no guns to lose or forget about. I was terrifically hungry.
In the morning it was bright, and they were sprinkling the streets of
the town, and we all had breakfast in a café. Bayonne is a nice town. It
is like a very clean Spanish town and it is on a big river. Already, so
early in the morning, it was very hot on the bridge across the river. We
walked out on the bridge and then took a walk through the town.

I was not at all sure Mike’s rods would come from Scotland in time, so
we hunted a tackle store and finally bought a rod for Bill up-stairs
over a drygoods store. The man who sold the tackle was out, and we had
to wait for him to come back. Finally he came in, and we bought a pretty
good rod cheap, and two landing-nets.

We went out into the street again and took a look at the cathedral. Cohn
made some remark about it being a very good example of something or
other, I forget what. It seemed like a nice cathedral, nice and dim,
like Spanish churches. Then we went up past the old fort and out to the
local Syndicat d’Initiative office, where the bus was supposed to start
from. There they told us the bus service did not start until the 1st of
July. We found out at the tourist office what we ought to pay for a
motor-car to Pamplona and hired one at a big garage just around the
corner from the Municipal Theatre for four hundred francs. The car was
to pick us up at the hotel in forty minutes, and we stopped at the café
on the square where we had eaten breakfast, and had a beer. It was hot,
but the town had a cool, fresh, early-morning smell and it was pleasant
sitting in the café. A breeze started to blow, and you could feel that
the air came from the sea. There were pigeons out in the square, and the
houses were a yellow, sun-baked color, and I did not want to leave the
café. But we had to go to the hotel to get our bags packed and pay the
bill. We paid for the beers, we matched and I think Cohn paid, and went
up to the hotel. It was only sixteen francs apiece for Bill and me, with
ten per cent added for the service, and we had the bags sent down and
waited for Robert Cohn. While we were waiting I saw a cockroach on the
parquet floor that must have been at least three inches long. I pointed
him out to Bill and then put my shoe on him. We agreed he must have just
come in from the garden. It was really an awfully clean hotel.

Cohn came down, finally, and we all went out to the car. It was a big,
closed car, with a driver in a white duster with blue collar and cuffs,
and we had him put the back of the car down. He piled in the bags and we
started off up the street and out of the town. We passed some lovely
gardens and had a good look back at the town, and then we were out in
the country, green and rolling, and the road climbing all the time. We
passed lots of Basques with oxen, or cattle, hauling carts along the
road, and nice farmhouses, low roofs, and all white-plastered. In the
Basque country the land all looks very rich and green and the houses and
villages look well-off and clean. Every village had a pelota court and
on some of them kids were playing in the hot sun. There were signs on
the walls of the churches saying it was forbidden to play pelota against
them, and the houses in the villages had red tiled roofs, and then the
road turned off and commenced to climb and we were going way up close
along a hillside, with a valley below and hills stretched off back
toward the sea. You couldn’t see the sea. It was too far away. You could
see only hills and more hills, and you knew where the sea was.

We crossed the Spanish frontier. There was a little stream and a bridge,
and Spanish carabineers, with patent-leather Bonaparte hats, and short
guns on their backs, on one side, and on the other fat Frenchmen in
kepis and mustaches. They only opened one bag and took the passports in
and looked at them. There was a general store and inn on each side of
the line. The chauffeur had to go in and fill out some papers about the
car and we got out and went over to the stream to see if there were any
trout. Bill tried to talk some Spanish to one of the carabineers, but it
did not go very well. Robert Cohn asked, pointing with his finger, if
there were any trout in the stream, and the carabineer said yes, but not
many.

I asked him if he ever fished, and he said no, that he didn’t care for
it.
"""

lawrence = """
I've got lodging in a bit of an old cottage in Engine Row very decent.
The man is engine-driver at High Park, tall, with a beard, and very
chapel. The woman is a birdy bit of a thing who loves anything
superior. King's English and allow-me! all the time. But they lost
their only son in the war, and it's sort of knocked a hole in them.
There's a long gawky lass of a daughter training for a school-teacher,
and I help her with her lessons sometimes, so we're quite the family.
But they're very decent people, and only too kind to me. I expect I'm
more coddled than you are.

I like farming all right. It's not inspiring, but then I don't ask to
be inspired. I'm used to horses, and cows, though they are very female,
have a soothing effect on me. When I sit with my head in her side,
milking, I feel very solaced. They have six rather fine Herefords.
Oat-harvest is just over and I enjoyed it, in spite of sore hands and a
lot of rain. I don't take much notice of people, but get on with them
all right. Most things one just ignores.

The pits are working badly; this is a colliery district like
Tevershall, only prettier. I sometimes sit in the Wellington and talk
to the men. They grumble a lot, but they're not going to alter
anything. As everybody says, the Notts-Derby miners have got their
hearts in the right place. But the rest of their anatomy must be in the
wrong place, in a world that has no use for them. I like them, but they
don't cheer me much: not enough of the old fighting-cock in them. They
talk a lot about nationalization, nationalization of royalties,
nationalization of the whole industry. But you can't nationalize coal
and leave all the other industries as they are. They talk about putting
coal to new uses, like Sir Clifford is trying to do. It may work here
and there, but not as a general thing, I doubt. Whatever you make
you've got to sell it. The men are very apathetic. They feel the whole
damned thing is doomed, and I believe it is. And they are doomed along
with it. Some of the young ones spout about a Soviet, but there's not
much conviction in them. There's no sort of conviction about anything,
except that it's all a muddle and a hole. Even under a Soviet you've
still got to sell coal: and that's the difficulty.

We've got this great industrial population, and they've got to be fed,
so the damn show has to be kept going somehow. The women talk a lot
more than the men, nowadays, and they are a sight more cock-sure. The
men are limp, they feel a doom somewhere, and they go about as if there
was nothing to be done. Anyhow, nobody knows what should be done in
spite of all the talk, the young ones get mad because they've no money
to spend. Their whole life depends on spending money, and now they've
got none to spend. That's our civilization and our education: bring up
the masses to depend entirely on spending money, and then the money
gives out. The pits are working two days, two and a half days a week,
and there's no sign of betterment even for the winter. It means a man
bringing up a family on twenty-five and thirty shillings. The women are
the maddest of all. But then they're the maddest for spending,
nowadays.

If you could only tell them that living and spending isn't the same
thing! But it's no good. If only they were educated to LIVE instead of
earn and spend, they could manage very happily on twenty-five
shillings. If the men wore scarlet trousers as I said, they wouldn't
think so much of money: if they could dance and hop and skip, and sing
and swagger and be handsome, they could do with very little cash. And
amuse the women themselves, and be amused by the women. They ought to
learn to be naked and handsome, and to sing in a mass and dance the old
group dances, and carve the stools they sit on, and embroider their own
emblems. Then they wouldn't need money. And that's the only way to
solve the industrial problem: train the people to be able to live and
live in handsomeness, without needing to spend. But you can't do it.
They're all one-track minds nowadays. Whereas the mass of people
oughtn't even to try to think, because they can't. They should be alive
and frisky, and acknowledge the great god Pan. He's the only god for
the masses, forever. The few can go in for higher cults if they like.
But let the mass be forever pagan.

But the colliers aren't pagan, far from it. They're a sad lot, a
deadened lot of men: dead to their women, dead to life. The young ones
scoot about on motor-bikes with girls, and jazz when they get a chance,
But they're very dead. And it needs money. Money poisons you when
you've got it, and starves you when you haven't.
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
She was the first “nice” girl he had ever known. In various unrevealed capacities he had come in contact with such people, but always with indiscernible barbed wire between. He found her excitingly desirable. He went to her house, at first with other officers from Camp Taylor, then alone. It amazed him—he had never been in such a beautiful house before. But what gave it an air of breathless intensity was that Daisy lived there—it was as casual a thing to her as his tent out at camp was to him. There was a ripe mystery about it, a hint of bedrooms up-stairs more beautiful and cool than other bedrooms, of gay and radiant activities taking place through its corridors, and of romances that were not musty and laid away already in lavender, but fresh and breathing and redolent of this year’s shining motor-cars and of dances whose flowers were scarcely withered. It excited him, too, that many men had already loved Daisy—it increased her value in his eyes. He felt their presence all about the house, pervading the air with the shades and echoes of still vibrant emotions.
But he knew that he was in Daisy’s house by a colossal accident. However glorious might be his future as Jay Gatsby, he was at present a penniless young man without a past, and at any moment the invisible cloak of his uniform might slip from his shoulders. So he made the most of his time. He took what he could get, ravenously and unscrupulously—eventually he took Daisy one still October night, took her because he had no real right to touch her hand.
He might have despised himself, for he had certainly taken her under false pretenses. I don’t mean that he had traded on his phantom millions, but he had deliberately given Daisy a sense of security; he let her believe that he was a person from much the same stratum as herself—that he was fully able to take care of her. As a matter of fact, he had no such facilities—he had no comfortable family standing behind him, and he was liable at the whim of an impersonal government to be blown anywhere about the world.
But he didn’t despise himself and it didn’t turn out as he had imagined. He had intended, probably, to take what he could and go—but now he found that he had committed himself to the following of a grail. He knew that Daisy was extraordinary, but he didn’t realize just how extraordinary a “nice” girl could be. She vanished into her rich house, into her rich, full life, leaving Gatsby—nothing. He felt married to her, that was all.
When they met again, two days later, it was Gatsby who was breathless, who was, somehow, betrayed. Her porch was bright with the bought luxury of star-shine; the wicker of the settee squeaked fashionably as she turned toward him and he kissed her curious and lovely mouth. She had caught a cold, and it made her voice huskier and more charming than ever, and Gatsby was overwhelmingly aware of the youth and mystery that wealth imprisons and preserves, of the freshness of many clothes, and of Daisy, gleaming like silver, safe and proud above the hot struggles of the poor.
“I can’t describe to you how surprised I was to find out I loved her, old sport. I even hoped for a while that she’d throw me over, but she didn’t, because she was in love with me too. She thought I knew a lot because I knew different things from her… Well, there I was, ’way off my ambitions, getting deeper in love every minute, and all of a sudden I didn’t care. What was the use of doing great things if I could have a better time telling her what I was going to do?”
On the last afternoon before he went abroad, he sat with Daisy in his arms for a long, silent time. It was a cold fall day, with fire in the room and her cheeks flushed. Now and then she moved and he changed his arm a little, and once he kissed her dark shining hair. The afternoon had made them tranquil for a while, as if to give them a deep memory for the long parting the next day promised. They had never been closer in their month of love, nor communicated more profoundly one with another, than when she brushed silent lips against his coat’s shoulder or when he touched the end of her fingers, gently, as though she were asleep.
He did extraordinarily well in the war. He was a captain before he went to the front, and following the Argonne battles he got his majority and the command of the divisional machine-guns. After the Armistice he tried frantically to get home, but some complication or misunderstanding sent him to Oxford instead. He was worried now—there was a quality of nervous despair in Daisy’s letters. She didn’t see why he couldn’t come. She was feeling the pressure of the world outside, and she wanted to see him and feel his presence beside her and be reassured that she was doing the right thing after all.
"""

Orwell = """
Mr Pitcairn does not tell us how and when it became clear that the P.O.U.M.
possessed scores of machine-guns and several thousand rifles.
I have given an estimate of the arms which were at three of the principal P.O.U.M.
buildings--about eighty rifles, a few bombs, and no machine-guns; i.e.
about sufficient for the armed guards which, at that time, all the political parties placed on their buildings.
It seems strange that afterwards, when the P.O.U.M.
was suppressed and all its
buildings seized, these thousands of weapons never came to light; especially the tanks and field-guns, which are not the kind of thing that can be hidden up the chimney.
But what is revealing in the two statements above is the complete ignorance they display of the local circumstances.
According to Mr Pitcairn the P.O.U.M.
stole tanks 'from the barracks'.
He does not tell us which barracks.
The P.O.U.M.
militiamen who were in Barcelona (now comparatively few, as direct
recruitment to the party militias had ceased) shared the Lenin Barracks with a considerably larger number of Popular Army troops.
Mr Pitcairn is asking us to believe, therefore, that the P.O.U.M.
stole tanks with the connivance of the Popular Army.
It is the same with the 'premises' on which the 75-mm.
guns were concealed.
There is no mention of where these 'premises' were.
Those batteries of guns, firing on the Plaza de Espana appeared in many newspaper reports, but I think we can say with certainty that they never existed.
As I mentioned earlier, I heard no artillery-fire during the fighting, though the Plaza de Espana was only a mile or so away.
A few days later I examined the Plaza de Espana and could find no buildings that showed marks of shell-fire.
And an eye-witness who was in that neighbourhood throughout the fighting declares that no guns ever appeared there.
(Incidentally, the tale of the stolen guns may have originated with Antonov-Ovseenko, the Russian Consul-General.
He, at any rate, communicated it to a well-known English journalist, who afterwards repeated it in good faith in a weekly paper.
Antonov-Ovseenko has since been 'purged'.
How this would affect his credibility I do not know.)
The truth is, of course, that these tales about tanks, field-guns, and so forth have only been invented because otherwise it is difficult to reconcile the scale of the Barcelona fighting with the P.O.U.M.
's small numbers.
It was necessary to claim that the P.O.U.M.
was wholly responsible for the fighting; it was also necessary to claim that it was an insignificant party with no following and 'numbered only a few thousand members', according to _Inprecor_.
The only hope of making both statements credible was to pretend that the P.O.U.M.
had all the weapons of a modern mechanized army.
But none of these projects ever comes anywhere near realization, and none
of the three super-states ever gains a significant lead on the others.
What is more remarkable is that all three powers already possess, in the
atomic bomb, a weapon far more powerful than any that their present
researches are likely to discover.
Although the Party, according to its habit, claims the invention for itself, atomic bombs first appeared as
early as the nineteen-forties, and were first used on a large scale about ten years later.
At that time some hundreds of bombs were dropped on industrial centres, chiefly in European Russia, Western Europe, and North America.
The effect was to convince the ruling groups of all countries that a few more atomic bombs would mean the end of organized society, and hence of their own power.
Thereafter, although no formal agreement was ever made or hinted at, no more bombs were dropped.
All three powers merely continue to produce atomic bombs and store them up against the decisive opportunity which they all believe will come sooner or later.
And meanwhile the art of war has remained almost stationary for thirty or forty years.
Helicopters are more used than they were formerly, bombing planes have been largely superseded by self-propelled projectiles, and the fragile movable battleship has given way to the almost unsinkable Floating
Fortress; but otherwise there has been little development.
The tank, the submarine, the torpedo, the machine gun, even the rifle and the hand grenade are still in use.
And in spite of the endless slaughters reported in the Press and on the telescreens, the desperate battles of earlier wars, in which hundreds of thousands or even millions of men were often killed in a few weeks, have never been repeated.
None of the three super-states ever attempts any manoeuvre which involves the risk of serious defeat.
When any large operation is undertaken, it is usually a surprise attack against an ally.
The strategy that all three powers are following, or pretend to themselves that they are following, is the same.
The plan is, by a combination of fighting, bargaining, and well-timed strokes of treachery, to acquire a ring of bases completely encircling one or other of the rival states, and then to sign a pact of friendship with that rival and remain on peaceful terms for so many years as to lull suspicion to sleep.
During this time rockets loaded with atomic bombs can be assembled at all the strategic spots; finally they will all be fired simultaneously, with effects so devastating as to make retaliation impossible.
It will then be time to sign a pact of friendship with the remaining world-power, in preparation for another attack.
This scheme, it is hardly necessary to say, is a mere daydream, impossible of realization.
Moreover, no fighting ever occurs except in the disputed areas round the Equator and the Pole: no invasion of enemy territory is ever undertaken.
This explains the fact that in some places the frontiers between the super-states are arbitrary.
Eurasia, for example, could easily conquer the British Isles, which are geographically part of Europe, or on the other hand it would be possible for Oceania to push its frontiers to the Rhine or even to the Vistula.
But this would violate the principle, followed on all sides though never formulated, of cultural integrity.
If Oceania were to conquer the areas that used once to be known as France and Germany, it would be necessary either to exterminate the inhabitants, a task of great physical difficulty, or to assimilate a population of about a hundred million people, who, so far as technical development goes, are roughly on
the Oceanic level.
The problem is the same for all three super-states.
It is absolutely necessary to their structure that there should be no contact with foreigners, except, to a limited extent, with war prisoners and coloured slaves.
Even the official ally of the moment is always regarded with the darkest suspicion.
War prisoners apart, the average citizen of Oceania never sets eyes on a citizen of either Eurasia or Eastasia, and he is forbidden the knowledge of foreign languages.
If he were allowed contact with foreigners he would discover that they are creatures similar to himself and that most of what he has been told about them is lies.
The sealed world in which he lives would be broken, and the fear, hatred, and self-righteousness on which his morale depends might evaporate.
It is therefore realized on all sides that however often Persia, or Egypt, or Java, or Ceylon may change hands, the main frontiers must never be crossed by anything except bombs.
"""

Kerouac = """
But in the morning (and I’m no Milarepa who could also sit naked in the snow and was seen flying on one occasion) here comes Ron Blake back with Pat McLear and Pat’s wife the beautiful one, and by God their little sweet five year old girl who is such a pleasant sight to see as she goes jongling and jiggling through the fields to look for flowers, everything to her is perfectly new beautiful primordial Garden of Eden morning here in this tortured human canyon — And a rather beautiful morning develops — There’s fog so we close the blinds and light the fire and the lamp, me and Pat, and sit there drinking from the jug he brought talking about literature and poetry while his wife listens and occasionally gets up to heat more coffee and tea or goes out to play with Ron and the little girl — Pat and I are in a serious talkative mood and I feel that lonely shiver in my chest which always warns me: you actually love people and you’re glad Pat is here.
Pat is one if not THE most handsome man I’ve ever seen — Strange that he’s announced in a preface to his poems that his heroes, his Triumvirate, are Jean Harlow, Rimbaud and Billy the Kid because he himself is handsome enough to play Billy the Kid in the movies, that same darkhaired handsome slightly sliteyed look you expect from the myth appearance of Billy the Kid (I suppose not the actual real life William Bonnie who’s said to’ve been a pimply cretin monster).
So we launch on a big discussion of everything in the comfortable gloom of the cabin by the warm red glow of the girly fire, I’m wearing dark glasses anyway for fun, Pat says “Well Jack I didnt have a chance to talk to you yesterday or even last year or even ten years ago when I first met you, I remember I was terrified of you and Pomeray when you ran up my steps one night with sticks of tea, you looked like a couple of car thieves or bank robbers — And you know a lot of this sneery stuff they’ve written against us, against San Francisco or beat poetry and writers is because a lot of us don’t LOOK like writers or intellectuals or anything, you and Pomeray I must say look awful in a way, I’m sure I dont fill the bill either”— ‘Man you oughta go to Hollywood and play Billy the Kid”— “Man I’d rather go to Hollywood and play Rimbaud”— ‘Well you can’t play Jean Harlow”— “I’d really like to just get my “Dark Brown” published in Paris, do you know that when you think it’s possible a word from you to Gallimard or Girodias would help”— “I dunno”— “Do you know that when I read your poems Mexico City Blues I immediately turned around and started writing a brand new way, you enlightened me with that book”— “But it’s nothing like what you do, in fact it’s miles away, I am a language spinner and you’re idea man” and so on we talk till about noon and Ron’s been in and out, “s’made jaunts to the beach with the little ladies and Pat and I don’t realize the sun has come out but still sit there deep in the cabin by now talking about Villon and Cervantes.
Suddenly, boom, the door of the cabin is flung open with a loud crash and a burst of sunlight illuminates the room and I see an Angel standing arm outstretched in the door! — It’s Cody! all dressed in his Sunday best in a suit! beside him are ranged several graduating golden angels from Evelyn golden beautiful wife down to the most dazzling angel of them all little Timmy with the sun striking off his hair in beams! - - It’s such an incredible sight and surprise that both Pat and I rise from our chairs involuntarily, like we’ve been lifted up in awe, or scared, tho I dont feel scared so much as ecstatically amazed as tho I’ve seen a vision… And the way Cody stands there not saying a word with his arm outstretched for some reason, struck a pose of some sort to surprise us or warn us, he’s so much like St Michael at the moment it’s unbelievable especially as I also suddenly realize what he’s just actually done, he’s had wife and kiddies sneak up ever so quiet up the porch steps (which are noisy and creaky), across the wood planks, easy and tiptoeing, stood there awhile while he prepared to fling the door open, all lined up and stood straight, then pow, he’s opened the door and thrown the golden universe into the dazzled mystic eyes of big hip Pat McLear and big amazed grateful me.
It reminds me of the time I once saw a whole tiptoeing gang of couples sneaking into our back kitchen door on West Street in Lowell the leader telling me to shush as I stand there nine years old amazed, then all bursting in on my father innocently listening to the Primo Carnera - Ernie Schaaft fight on the old 1930s radio — For a big roaring toot… But Cody’s oldfashioned family tiptoe sneak carries that strange apocalyptic burst of gold he somehow always manages to produce, like I said elsewhere the time in Mexico he drove an old car over a rutted road very slowly as we were all high on tea and I saw golden Heaven, or the other times he’s always seemed so golden like as I say in a davenport of some sort in Heaven in the golden top of Heaven.
Not that he means to produce this effect: he’s just standing there with innate dramatic mystery holding forth his arm as if to say Behold, the sun! and Behold, the angels! sorta pointing at all the golden heads of his family and Pat and I stand aghast.
I love Nobody was paying any attention to me up there.
I should have known better.
It was Terry who brought my soul back; on the tent stove she warmed up the food, and it was one of the greatest meals of my life, I was so hungry and tired.
Sighing like an old Negro cotton-picker, I reclined on the bed and smoked a cigarette.
Dogs barked in the cool night. Rickey and Ponzo had given up calling in the evenings.
I was satisfied with that. Terry curled up beside me, Johnny sat on my chest, and they drew pictures of animals in my notebook.
The light of our tent burned on the frightful plain.
The cowboy music twanged in the roadhouse and carried across the fields, all sadness.
It was all right with me. I kissed my baby and we put out the lights.
In the morning the dew made the tent sag; I got up with my towel and toothbrush and went to the general motel toilet to wash; then I came back, put on my pants, which were all torn from kneeling in the earth and had been sewed by Terry in the evening, put on my ragged straw hat, which had originally served as Johnny’s toy hat, and went across the highway with my canvas cotton-bag.
Every day I earned approximately a dollar and a half.
It was just enough to buy groceries in the evening on the bicycle.
The days rolled by. I forgot all about the East and all about Dean and Carlo and the bloody road.
Johnny and I played all the time; he liked me to throw him up in the air and down in the bed.
Terry sat mending clothes.
I was a man of the earth, precisely as I had dreamed I would be, in Paterson.
There was talk that Terry’s husband was back in Sabinal and out for me; I was ready for him.
One night the Okies went mad in the roadhouse and tied a man to a tree and beat him to a pulp with sticks.
I was asleep at the time and only heard about it.
From then on I carried a big stick with me in the tent in case they got the idea we Mexicans were fouling up their trailer camp.
They thought I was a Mexican, of course; and in a way I am.
But now it was October and getting much colder in the nights.
The Okie family had a woodstove and planned to stay for the winter.
We had nothing, and besides the rent for the tent was due. Terry and I bitterly decided we’d have to leave.
“Go back to your family,” I said.
“For God’s sake, you can’t be batting around tents with a baby like Johnny; the poor little tyke is cold.”
Terry cried because I was criticizing her motherly instincts; I meant no such thing.
When Ponzo came in the truck one gray afternoon we decided to see her family about the situation.
But I mustn’t be seen and would have to hide in the vineyard.
We started for Sabinal; the truck broke down, and simultaneously it started to rain wildly.
“We sat in the old truck, cursing. Ponzo got out and toiled in the rain.”
He was a good old guy after all. We promised each other one more big bat.
Off we went to a rickety bar in Sabinal Mextown and spent an hour sopping up the brew.
I was through with my chores in the cottonfield. I could feel the pull of my own life calling me back.
I shot my aunt a penny postcard across the land and asked for another fifty.
We drove to Terry’s family’s shack. It was situated on the old road that ran between the vineyards.
It was dark when we got there. They left me off a quarter-mile away and drove to the door.
Light poured out of the door; Terry’s six other brothers were playing their guitars and singing.
The old man was drinking wine. I heard shouts and arguments above the singing.
They called her a whore because she’d left her no-good husband and gone to LA and left Johnny with them.
The old man was yelling. But the sad, fat brown mother prevailed, as she always does among the great fellahin peoples of the world, and Terry was allowed to come back home.
The brothers began to sing gay songs, fast.
I huddled in the cold, rainy wind and watched everything across the sad vineyards of October in the valley. 
It’s not the words so much as their great harmonic tune and the way Billie sings it, like a woman stroking her man’s hair in soft lamplight. The winds howled.
"Okay I will."
Japhy, kneeling there studying his star map, leaning forward slightly to peek up through the overhanging gnarled old rock country trees, with his goatee and all, looked, with thatmighty graw-faced rock behind him, like, exactly like the vision I had of the old Zen Masters of China out in the wilderness.
He was leaning forward on his knees, upward looking, as if with a holy sutra in his hands.
Pretty soon he went to the snowbank and brought back thechocolate pudding which was now ice cold and absolutely delicious beyond words.
We ate it all up.
"Maybe we oughta leave some for Morley."
"Ah it won't keep, it'll melt in the morning sun."
As the fire stopped roaring and just got to be red coals, but big ones six feet long, the night interposed its icy crystal feel 1 more and more but with the smell of smoking logs it was as delicious as chocolate pudding.
For a while I went on a little walk by myself, out by the shallow iced creek, and sat meditating against a stump of dirt and the huge mountain walls on both sides of our valley were silent masses.
Too cold to do this more than a minute.
As I came back our orange fire casting its glow on the big rock, and Japhy kneeling and peering up at the sky, and all of it ten thousand feet above the gnashing world, was a picture of peace and good sense.
There was another aspect of Japhy that amazed me: his tremendous and tender sense of charity.
He was always giving things, always practicing what the Buddhists call the Paramita of Dana, the perfection of charity.
Now when I came back and sat down by the fire he said "Well Smith it's about time you owned a set of juju beads you can have these," and he handed me the brown wood beads run together over a strong string with the string, black and shiny, coming out at the large bead at the end in a pretty loop.
"Aw you can't give me something like this, these things come from Japan don't they?"
"I've got another set of black ones.
Smith that prayer you gave me tonight is worth that set of juju beads, but you can have it anyway."
A few minutes later he cleaned out the rest of the chocolate pudding but made sure that I got most of it.
Then when he laid boughs over the rock of our clearing and the poncho over that he made sure his sleeping bag was farther away from the fire than mine so I would sure to be warm.
He was always practicing charity.
In fact he taught me, and a week later I was giving him nice new undershirts I'd discovered in the Goodwill store.
He'd turn right around and make me a gift of a plastic container to keep food in.
For a joke I'd give him a gift of a huge flower from Alvah's yard.
Solemnly a day later he'd bring me a little bouquet of flowers picked in the street plots of Berkeley.
"And you can keep the sneakers too," he said.
"I've got another pair older than those but just as good."
"Aw I can't be taking all your things."
"Smith you don't realize it's a privilege to practice giving presents to others."
The way he did it was charming; there was nothing glittery and Christmasy about it, but almost sad, and
sometimes his gifts were old beat-up things but they had the charm of usefulness and sadness of his giving.
"""

Austen = """
Mr. Collins, said she, speaks highly both of Lady Catherine and her daughter; but, from some particulars that he has related of her Ladyship, I suspect his gratitude misleads him; and that, in spite of her being his patroness, she is an arrogant, conceited woman.
I believe her to be both in a great degree, replied Wickham; I have not seen her for many years; but I very well remember that I never liked her, and that her manners were dictatorial and insolent.
She has the reputation of being remarkably sensible and clever; but I rather believe she derives part of her abilities from her rank and fortune, part from her authoritative manner, and the rest from the pride of her nephew, who chooses that everyone connected with him should have an understanding of the first class.
Elizabeth allowed that he had given a very rational account of it, and they continued talking together with mutual satisfaction till supper put an end to cards, and gave the rest of the ladies their share of Mr. Wickham’s attentions.
There could be no conversation in the noise of Mrs. Philips’s supper party, but his manners recommended him to everybody.
Whatever he said, was said well; and whatever he did, done gracefully.
Elizabeth went away with her head full of him.
She could think of nothing but of Mr. Wickham, and of what he had told her, all the way home; but there was not time for her even to mention his name as they went, for neither Lydia nor Mr. Collins were once silent.
Lydia talked incessantly of lottery tickets, of the fish she had lost and the fish she had won; and Mr. Collins, in describing the civility of Mr. and Mrs. Philips, protesting that he did not in the least regard his losses at whist, enumerating all the dishes at supper, and repeatedly fearing that he crowded his cousins, had more to say than he could well manage before the carriage stopped at Longbourn House.
The situation of the house was good.
High hills rose immediately behind, and at no great distance on each side; some of which were open downs, the others cultivated and woody.
The village of Barton was chiefly on one of these hills, and formed a pleasant view from the cottage windows.
The prospect in front was more extensive; it commanded the whole of the valley, and reached into the country beyond.
The hills which surrounded the cottage terminated the valley in that direction; under another name, and in another course, it branched out again between two of the steepest of them.
With the size and furniture of the house Mrs. Dashwood was upon the whole well satisfied; for though her former style of life rendered many additions to the latter indispensable, yet to add and improve was a delight to her; and she had at this time ready money enough to supply all that was wanted of greater elegance to the apartments.
As for the house itself, to be sure, said she, it is too small for our family, but we will make ourselves tolerably comfortable for the present, as it is too late in the year for improvements.
Perhaps in the spring, if I have plenty of money, as I dare say I shall, we may think about building.
These parlors are both too small for such parties of our friends as I hope to see often collected here; and I have some thoughts of throwing the passage into one of them with perhaps a part of the other, and so leave the remainder of that other for an entrance; this, with a new drawing room which may be easily added, and a bed-chamber and garret above, will make it a very snug little cottage.
I could wish the stairs were handsome.
But one must not expect every thing; though I suppose it would be no difficult matter to widen them.
I shall see how much I am before-hand with the world in the spring, and we will plan our improvements accordingly.
In the mean time, till all these alterations could be made from the savings of an income of five hundred a-year by a woman who never saved in her life, they were wise enough to be contented with the house as it was; and each of them was busy in arranging their particular concerns, and endeavoring, by placing around them books and other possessions, to form themselves a home.
Marianne’s pianoforte was unpacked and properly disposed of; and Elinor’s drawings were affixed to the walls of their sitting room.
The Crawfords, without wanting to be cured, were very willing to stay.
Mary was satisfied with the Parsonage as a present home, and Henry equally ready to lengthen his visit.
He had come, intending to spend only a few days with them; but Mansfield promised well, and there was nothing to call him elsewhere.
It delighted Mrs. Grant to keep them both with her, and Dr. Grant was exceedingly well contented to have it so: a talking pretty young woman like Miss Crawford is always pleasant society to an indolent, stay-at-home man; and Mr. Crawford’s being his guest was an excuse for drinking claret every day.
The Miss Bertrams’ admiration of Mr. Crawford was more rapturous than anything which Miss Crawford’s habits made her likely to feel.
She acknowledged, however, that the Mr. Bertrams were very fine young men, that two such young men were not often seen together even in London, and that their manners, particularly those of the eldest, were very good.
He had been much in London, and had more liveliness and gallantry than Edmund, and must, therefore, be preferred; and, indeed, his being the eldest was another strong claim.
She had felt an early presentiment that she should like the eldest best. She knew it was her way.
Tom Bertram must have been thought pleasant, indeed, at any rate; he was the sort of young man to be generally liked, his agreeableness was of the kind to be oftener found agreeable than some endowments of a higher stamp, for he had easy manners, excellent spirits, a large acquaintance, and a great deal to say; and the reversion of Mansfield Park, and a baronetcy, did no harm to all this.
Miss Crawford soon felt that he and his situation might do.
She looked about her with due consideration, and found almost everything in his favour: a park, a real park, five miles round, a spacious modern-built house, so well placed and well screened as to deserve to be in any collection of engravings of gentlemen’s seats in the kingdom, and wanting only to be completely new furnished—pleasant sisters, a quiet mother, and an agreeable man himself—with the advantage of being tied up from much gaming at present by a promise to his father, and of being Sir Thomas hereafter.
It might do very well; she believed she should accept him; and she began accordingly to interest herself a little about the horse which he had to run at the B—— races.
"""

Faulkner = """
The day dawned bleak and chill, a moving wall of gray light out of the northeast which, instead of dissolving into moisture, seemed to disintegrate into minute and venomous particles, like dust that, when Dilsey opened the door of the cabin and emerged, needled laterally into her flesh, precipitating not so much a moisture as a substance partaking of the quality of thin, not quite congealed oil. She wore a stiff black straw hat perched upon her turban, and a maroon velvet cape with a border of mangy and anonymous fur above a dress of purple silk, and she stood in the door for a while with her myriad and sunken face lifted to the weather, and one gaunt hand flac-soled as the belly of a fish, then she moved the cape aside and examined the bosom of her gown.

The gown fell gauntly from her shoulders, across her fallen breasts, then tightened upon her paunch and fell again, ballooning a little above the nether garments which she would remove layer by layer as the spring accomplished and the warm days, in color regal and moribund. She had been a big woman once but now her skeleton rose, draped loosely in unpadded skin that tightened again upon a paunch almost dropsical, as though muscle and tissue had been courage or fortitude which the days or the years had consumed until only the indomitable skeleton was left rising like a ruin or a landmark above the somnolent and impervious guts, and above that the collapsed face that gave the impression of the bones themselves being outside the flesh, lifted into the driving day with an expression at once fatalistic and of a child's astonished disappointment, until she turned and entered the house again and closed the door.

The earth immediately about the door was bare. It had a patina, as though from the soles of bare feet in generations, like old silver or the walls of Mexican houses which have been plastered by hand. Beside the house, shading it in summer, stood three mulberry trees, the fledged leaves that would later be broad and placid as the palms of hands streaming flatly undulant upon the driving air. A pair of jaybirds came up from nowhere, whirled up on the blast like gaudy scraps of cloth or paper and lodged in the mulberries, where they swung in raucous tilt and recover, screaming into the wind that ripped their harsh cries onward and away like scraps of paper or of cloth in turn. Then three more joined them and they swung and tilted in the wrung branches for a time, screaming. The door of the cabin opened and Dilsey emerged once more, this time in a man's felt hat and an army overcoat, beneath the frayed skirts of which her blue gingham dress fell in uneven balloonings, streaming too about her as she crossed the yard and mounted the steps to the kitchen door.

A moment later she emerged, carrying an open umbrella now, which she slanted ahead into the wind, and crossed to the woodpile and laid the umbrella down, still open. Immediately she caught at it and arrested it and held to it for a while, looking about her. Then she closed it and laid it down and stacked stovewood into her crooked arm, against her breast, and picked up the umbrella and got it open at last and returned to the steps and held the wood precariously balanced while she contrived to close the umbrella, which she propped in the corner just within the door. She dumped the wood into the box behind the stove. Then she removed the overcoat and hat and took a soiled apron down from the wall and put it on and built a fire in the stove. While she was doing so, rattling the grate bars and clattering the lids, Mrs Compson began to call her from the head of the stairs.

She wore a dressing gown of quilted black satin, holding it close under her chin. In the other hand she held a red rubber hot water bottle and she stood at the head of the back stairway, calling "Dilsey" at steady and inflectionless intervals into the quiet stairwell that descended into complete darkness, then opened again where a gray window fell across it. "Dilsey," she called, without inflection or emphasis or haste, as though she were not listening for a reply at all. "Dilsey."
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

combined_df.to_csv("sentences1.csv")