# this file is used to clean the queries in the jsonl file
import json
import re
import html
import unicodedata

def clean_query_text(text):
    if not isinstance(text, str):
        return text
    
    # Step 1: HTML unescape (handles &quot;, &amp;, etc.)
    text = html.unescape(text)
    
    # Step 2: More aggressive escape sequence removal
    text = text.replace('\\"', '"')    # Unescape quotes
    text = text.replace("\\'", "'")    # Unescape apostrophes
    text = text.replace('\\n', ' ')    # Replace newlines with spaces
    text = text.replace('\\t', ' ')    # Replace tabs with spaces
    text = text.replace('\\r', ' ')    # Replace carriage returns
    text = text.replace('\\/', '/')    # Unescape forward slashes
    text = text.replace('\\\\', '\\')  # Double backslashes to single
    
    # Step 3: Remove ALL remaining backslash escapes - be more aggressive
    text = re.sub(r'\\(.)', r'\1', text)  # Remove backslash before any character
    text = re.sub(r'\\', '', text)        # Remove any remaining lone backslashes
    
    # Step 4: Handle any remaining quote issues
    text = re.sub(r'"{2,}', '"', text)    # Multiple quotes to single
    
    # Step 5: Remove obvious artifacts only
    text = re.sub(r'```+', '', text)   # Remove ``` markers
    
    # Step 7: Normalize Unicode characters (conservative normalization)
    text = unicodedata.normalize('NFKC', text)
    
    # Step 8: Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)   # Multiple whitespace to single space
    text = text.strip()                # Remove leading/trailing whitespace
    
    return text

def clean_jsonl_queries(input_file, output_file):
    cleaned_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON
                obj = json.loads(line.strip())
                total_count += 1
                
                # Clean the query field if it exists
                if 'query' in obj and obj['query']:
                    original_query = obj['query']
                    cleaned_query = clean_query_text(original_query)
                    
                    if cleaned_query != original_query:
                        obj['query'] = cleaned_query
                        cleaned_count += 1
                        if line_num <= 5:  # Show first 5 changes for verification
                            print(f"Line {line_num}: Cleaned query")
                            print(f"  Before: {original_query[:100]}...")
                            print(f"  After:  {cleaned_query[:100]}...")
                            print()
                
                # Write cleaned object
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                # Write original line if JSON parsing fails
                outfile.write(line)
    
    print(f"✅ Processing complete!")
    print(f"📊 Total records: {total_count}")
    print(f"🧹 Queries cleaned: {cleaned_count}")
    print(f"💾 Output saved to: {output_file}")

if __name__ == "__main__":
    input_file = ""
    output_file = ""
    
    print("🚀 Starting minimal query cleanup...")
    clean_jsonl_queries(input_file, output_file)