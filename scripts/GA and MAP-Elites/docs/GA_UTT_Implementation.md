1. What is being evolved (chromosome = one UTT)
A chromosome is one full UTT: it encodes the numbers that define your game (unit stats, costs, etc.).
So “one individual” = one UTT = one possible game balance.
Typical contents (from your ga_chromosome / config):
Unit types: cost, HP, move speed, attack, range, produce cost, etc.
Can also include game-level parameters (e.g. harvest amount, resource amounts) if your chromosome includes them.
So the GA is literally evolving the UTT parameters; each chromosome is a candidate UTT.
2. How a UTT is tested (fitness)
For each chromosome (candidate UTT):
UTT file
The chromosome is turned into a UTT (e.g. JSON) and written to a file (e.g. test_utt.json).
Games
That UTT is loaded into MicroRTS, and you run AI vs AI games (e.g. LightRush vs WorkerRush) on a fixed map.
So: same AIs, same map, different UTT each time.
Outcomes → fitness
From the match results (wins/losses/draws, and possibly game length), the code computes:
Balance – e.g. how close to 50–50 the matchup is (more balanced → better).
Duration – e.g. not too many very short or very long games.
Diversity – e.g. variety of outcomes (less “always same result” → better when you have multiple matchups).
Single number
These are combined into one fitness value, e.g.
fitness = α·balance + β·duration + γ·diversity.
Higher fitness = better UTT for your goals.
So: generate UTT from chromosome → run games with that UTT → score the results → that’s the fitness for that UTT.
3. Evolution loop (how new UTTs are generated)
Population: You keep a set of N chromosomes (N UTTs), e.g. 20.
Generations: You repeat the following:
Evaluate
For each chromosome, build its UTT, run the games, compute fitness (as above).
Select
Keep the best ones (e.g. elitism) and choose parents for the next generation (e.g. tournament selection: pick among the better individuals).
Crossover
Create child UTTs by mixing two parents (e.g. take some parameters from parent A, some from parent B). So new UTTs are combinations of existing good UTTs.
Mutation
Randomly tweak some parameters in the children (e.g. change a cost or HP by a small amount). So you get new UTTs that are small variations of current ones.
Replace
The new children (plus maybe the best old ones) become the next population. Then go back to “Evaluate” for the next generation.
So over generations you get:
Selection → keep better UTTs.
Crossover → combine good UTTs into new ones.
Mutation → explore nearby UTTs.
The GA is not writing the UTT by hand; it’s searching over the space of UTT parameters by evolving chromosomes that encode those parameters, and using game outcomes (wins/losses/draws, duration) as the signal for what “good” means.
4. Short summary
Step	What happens
Chromosome	Encodes one UTT (all the numbers that define units/game).
Fitness	Build that UTT → run AI vs AI games with it → score balance/duration/diversity → one fitness value.
Evolution	Select good UTTs → crossover (mix two UTTs) → mutate (small random changes) → new population of UTTs.
Result	Over generations, the population tends to contain UTTs that give more balanced, better-lasting, more varied games (according to your fitness weights).
So in one sentence: the GA generates UTTs by evolving a population of candidate UTTs (chromosomes), evaluates each by running real MicroRTS games with that UTT, and uses selection, crossover, and mutation to produce new UTTs that improve balance and the other objectives you defined.