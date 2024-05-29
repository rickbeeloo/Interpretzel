import wget
import csv 
from collections import defaultdict

# Download iso table
url = "https://bacdive.dsmz.de/isolation-sources/csv?filter-domain=&filter-phylum=&filter-class=&filter-ordo=&filter-family=&filter-genus=&iso_3_country=&polygon-strain-ids=&sort_by_order=ASC&sort_by=st.species&pfc=&csv=1"
filename = wget.download(url)

# Read in reader 
per_id = defaultdict(list)
with open(filename, "r") as in_file:
    reader = csv.DictReader(in_file, delimiter = ",")
    # First we gather everything per ID as they span multiple lines
    # Only the first occurnce of the ID has the iso info, the others 
    # with the same ID are empty 
    for row in reader:
        per_id[row["ID"]].append(row)

# Now per ID get:
# - iso text 
# - tags 
source_tags = defaultdict(set)
untagged = 0
for id, rows in per_id.items():
    source = ""
    tags = []
    for row in rows:
        iso = row["Isolation source"].strip()
        if iso:
            source = iso 
        cat1 = row["Category 1"]
        cat2 = row["Category 2"]
        cat3 = row["Category 3"]
        tags.extend([cat1, cat2, cat3])
    tags = [tag for tag in tags if tag.strip()]
    if len(tags) > 0:
        source_tags[source].update(tags)
    else:
        untagged += 1


# Create a benchmark file
with open("bacdive_expected.tsv", "w") as out: 
    out.write("source\ttags\n")
    for source, tags in source_tags.items():
       tags = ", ".join(sorted(list(tags)))
       out.write(f"{source}\t{tags}\n")

# Load categories that we will test with (those also in SPIRE)
in_spire = wget.download("https://raw.githubusercontent.com/rickbeeloo/FAIRLLM/main/supp_table_S1_SPIRE_vs_BacDive.csv")
with open(in_spire, "r") as in_file:
    reader = csv.DictReader(in_file, delimiter="\t")
    targets = [row["BacDive tag"].strip() for row in reader if row["BacDive tag"].strip() ]

# Create a class generation file with two examples
# i.e. two-shot
import random 
random.seed(123) 
with open('bacdive_examples.tsv', "w") as out:
    per_tag = defaultdict(list)

    for source, tags in source_tags.items():
       for tag in tags:
           per_tag[tag].append(source)

    out.write("class_name\texamples\n")
    for class_name, examples in per_tag.items():
        if class_name in targets: # Only those also in SPIRE
            try:
                random_examples =  ", ".join(random.sample(examples, 2))
                out.write(f"{class_name}\t{random_examples}\n")
            except ValueError:
                print(f"Failed to sample for: {class_name}")
     

print("Untagged bacdive samples: ", untagged)