sed -e 's/numeric//g' feature-spec.txt  | tr ' ' '\n' | sort | uniq > feature-base-words.txt


