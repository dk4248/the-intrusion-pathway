"""
60 irrelevant generic-knowledge sentences, used as the candidate pool
for the 'irrelevant' context class. For each sample, stage_f_merge.py picks
the sentence with the most negative mean Δ_gold (gold-logit shift measured by
stage_e_irrelevant_screen.py, i.e. the largest damage to gold) across 4 models.

Sentences deliberately avoid:
  - named countries, capitals, cities, landmarks, continents
  - currencies, languages
  - named fruits + colors
  - musical instruments, named people
so they don't accidentally overlap with any of our 10 LRE relation domains.
"""

IRRELEVANT_POOL = [
    "Diamonds are formed deep under immense pressure.",
    "Butter is traditionally made from cream.",
    "Bees produce honey inside the hive.",
    "Clouds are made of tiny water droplets.",
    "Coffee is brewed from roasted beans.",
    "Tea leaves are steeped in hot water.",
    "Candles are usually made of wax.",
    "Paper is typically made from wood pulp.",
    "Soap is made by mixing fats with an alkaline solution.",
    "Glass is produced by melting sand at high temperatures.",
    "Iron rusts when exposed to moisture and oxygen.",
    "Rain falls from clouds to the ground.",
    "Lightning strikes from the sky during storms.",
    "Snowflakes form in cold atmospheric conditions.",
    "Rainbows appear when sunlight refracts through water droplets.",
    "Fog reduces visibility on roads in the morning.",
    "Wind is caused by differences in air pressure.",
    "Thunder follows lightning during a storm.",
    "Water boils at one hundred degrees Celsius.",
    "Ice melts when the temperature rises above freezing.",
    "Steam rises from boiling water.",
    "Salt dissolves quickly in warm water.",
    "Sugar dissolves faster when the water is stirred.",
    "Oil floats on top of water due to its lower density.",
    "Flour is ground from grain in a mill.",
    "Bread rises when yeast produces carbon dioxide.",
    "Cheese is aged for various lengths of time.",
    "Yogurt is produced by fermenting milk with bacteria.",
    "Eggs harden when they are boiled.",
    "Meat is cooked to ensure it is safe to eat.",
    "Leather is produced by tanning animal hides.",
    "Wool is sheared from sheep once a year.",
    "Silk is spun by silkworms inside their cocoons.",
    "Cotton is harvested from the cotton plant.",
    "Rubber is derived from the sap of certain trees.",
    "Plastic is molded after being heated to a liquid state.",
    "Metals expand slightly when they are heated.",
    "Magnets attract materials that contain iron.",
    "Electricity flows through copper wires.",
    "Batteries store energy in a chemical form.",
    "Engines convert fuel into mechanical motion.",
    "Gears transmit rotational force between shafts.",
    "Pulleys help lift heavy loads with less effort.",
    "Levers multiply the force applied at one end.",
    "Wheels reduce friction during transportation.",
    "Springs compress under load and return to shape.",
    "Ropes are braided from many smaller strands.",
    "Knots are used to secure objects together.",
    "Nails hold wooden boards together in construction.",
    "Screws have spiral threads along their shaft.",
    "Hammers drive nails into wood with impact force.",
    "Saws cut through material using sharp teeth.",
    "Drills make holes by spinning a sharp bit.",
    "Pencils leave marks on paper through friction.",
    "Erasers remove pencil marks from paper.",
    "Scissors cut paper between two sharpened blades.",
    "Glue forms a strong bond as it dries.",
    "Tape sticks to smooth surfaces very well.",
    "Books are bound along one edge for reading.",
    "Mirrors reflect light back toward the viewer.",
]

if __name__ == "__main__":
    print(f"Irrelevant pool size: {len(IRRELEVANT_POOL)}")
    for i, s in enumerate(IRRELEVANT_POOL):
        print(f"{i:2d}. {s}")
