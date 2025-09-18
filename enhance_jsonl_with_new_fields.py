import json
import random
from collections import defaultdict
from typing import Dict, List, Set

def load_and_analyze_data(input_file: str) -> tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    print(f"Loading and analyzing data from: {input_file}")
    
    subreddit_entities = defaultdict(set)
    subreddit_entity_lists = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                subreddit = record.get('source_subreddit', 'Unknown')
                entities = record.get('combined_comment_entities', [])
                
                # Add entities to subreddit groups
                for entity in entities:
                    if isinstance(entity, str) and entity.strip():
                        entity_original = entity.strip()
                        subreddit_entities[subreddit].add(entity_original)
                        subreddit_entity_lists[subreddit].append(entity_original)
                
                if line_num % 1000 == 0:
                    print(f"  Processed {line_num:,} records...")
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error on line {line_num}: {e}")
                continue
    
    print(f"  Completed analysis of {line_num:,} records")
    print(f"  Found {len(subreddit_entities)} subreddits")
    
    return subreddit_entities, subreddit_entity_lists

def create_limited_gt(ground_truth: List[str]) -> List[str]:
    if not ground_truth:
        return []
    
    seen_lower = set()
    unique_entities = []
    
    for entity in ground_truth:
        if isinstance(entity, str) and entity.strip():
            entity_original = entity.strip()
            entity_lower = entity_original.lower()
            
            if entity_lower and entity_lower not in seen_lower:
                seen_lower.add(entity_lower)
                unique_entities.append(entity_original)
    
    # Return up to 10 unique entities
    return unique_entities[:10]

def create_final_candidates(subreddit: str, subreddit_entity_lists: Dict[str, List[str]], limited_gt: List[str]) -> List[str]:
    if subreddit not in subreddit_entity_lists:
        return limited_gt.copy() if limited_gt else []
    
    # Deduplicate entities using case-insensitive comparison while preserving original casing
    seen_lower = set()
    unique_entities = []
    
    for entity in subreddit_entity_lists[subreddit]:
        entity_lower = entity.lower()
        if entity_lower not in seen_lower:
            seen_lower.add(entity_lower)
            unique_entities.append(entity)
    
    # Start with all limited_gt entities
    final_candidates = limited_gt.copy() if limited_gt else []
    
    # Find entities that are not already in limited_gt (case-insensitive comparison)
    limited_gt_lower = {entity.lower() for entity in limited_gt} if limited_gt else set()
    remaining_entities = [entity for entity in unique_entities if entity.lower() not in limited_gt_lower]
    
    # Randomly select up to 40 additional entities
    if remaining_entities:
        num_to_add = min(90, len(remaining_entities))
        additional_entities = random.sample(remaining_entities, num_to_add)
        final_candidates.extend(additional_entities)
    
    # Shuffle the final candidates list to randomize order
    random.shuffle(final_candidates)
    
    return final_candidates

def enhance_jsonl_file(input_file: str, output_file: str):
    print("=" * 60)
    print("ENHANCING JSONL FILE WITH NEW FIELDS")
    print("=" * 60)
    
    # Step 1: Load and analyze data
    print("\nStep 1: Loading and analyzing data...")
    subreddit_entities, subreddit_entity_lists = load_and_analyze_data(input_file)
    
    # Step 2: Process and enhance records
    print("\nStep 2: Processing and enhancing records...")
    enhanced_records = 0
    records_with_limited_gt = 0
    records_with_final_candidates = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line.strip())
                enhanced_records += 1
                
                # Add limited_gt field
                ground_truth = record.get('ground_truth', [])
                limited_gt = create_limited_gt(ground_truth)
                record['limited_gt'] = limited_gt
                if limited_gt:
                    records_with_limited_gt += 1
                
                # Add final_candidates field
                subreddit = record.get('source_subreddit', 'Unknown')
                final_candidates = create_final_candidates(subreddit, subreddit_entity_lists, limited_gt)
                record['final_candidates'] = final_candidates
                if final_candidates:
                    records_with_final_candidates += 1
                
                # Write enhanced record
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if line_num % 1000 == 0:
                    print(f"  Enhanced {line_num:,} records...")
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
                continue
    
    # Step 3: Print summary
    print("\n" + "=" * 60)
    print("ENHANCEMENT COMPLETE")
    print("=" * 60)
    print(f"Total records processed: {enhanced_records:,}")
    print(f"Records with limited_gt: {records_with_limited_gt:,}")
    print(f"Records with final_candidates: {records_with_final_candidates:,}")
    print(f"Output file: {output_file}")
    
    # Print subreddit statistics
    print(f"\nSubreddit entity counts:")
    for subreddit, entities in sorted(subreddit_entities.items()):
        print(f"  {subreddit:20s}: {len(entities):,} unique entities")

def main():
    input_file = ""
    output_file = ""
    
    enhance_jsonl_file(input_file, output_file)
    print("\nScript completed successfully!")
        


if __name__ == "__main__":
    main()
