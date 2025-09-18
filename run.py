import csv
import pandas as pd
import json
import os
import argparse
from typing import List, Dict, Any
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def format_transcript(csv_file_path: str, prompt_type: str = "all") -> str:
    """
    Format a classroom transcript CSV with teaching practice labels and instance IDs.
    Groups lines by bubble_idx and organizes instances under corresponding turns.
    
    Args:
        csv_file_path: Path to the CSV transcript file
        prompt_type: Type of prompt ("otr", "praise", or "all")
        
    Returns:
        Formatted transcript string with teaching practice annotations and instance IDs
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        formatted_lines = []
        
        # Group by bubble_idx
        for bubble_idx, group in df.groupby('bubble_idx'):
            # Skip empty bubbles
            if group.empty:
                continue
                
            # Get the speaker for this bubble (should be consistent within a bubble)
            speaker = group.iloc[0]['speaker']
            
            # Combine all utterances in this bubble
            utterances = []
            bubble_instances = []
            instance_utterances = {}  # Track utterances by instance key
            
            for _, row in group.iterrows():
                utterance = row['utterance']
                duration = row.get('duration_in_seconds', 0)
                
                # Handle silence speakers specially
                if speaker == 'silence':
                    utterance = f'[silence for {duration:.1f} seconds]'
                    utterances.append(utterance)
                else:
                    # Handle untranscribed utterances for non-silence speakers
                    if pd.isna(utterance) or utterance == '' or utterance == '[Untranscribed]':
                        utterance = f'[{speaker} talk for {duration:.1f} seconds]'
                        utterances.append(utterance)
                    else:
                        # Add transcribed utterances
                        utterances.append(utterance)
                
                # Extract teaching practice instances based on prompt type
                # Use umbrella instance IDs and consolidate multiple labels
                if prompt_type == "otr" or prompt_type == "all":
                    # Check for OTR umbrella instance
                    if row.get('opportunities_to_respond') == 1 and row.get('opportunities_to_respond_instance_id', 0) > 0:
                        instance_id = int(row.get('opportunities_to_respond_instance_id', 0))
                        instance_key = f"INSTANCE #{instance_id}"
                        
                        # Collect all practice labels for this instance
                        practices = []
                        if row.get('opportunities_to_respond') == 1:
                            practices.append("OTR")
                        if row.get('open_ended_questions') == 1:
                            practices.append("OEQ")
                        if row.get('presses_for_explanation') == 1:
                            practices.append("PRESS")
                        if row.get('wait_times') == 1:
                            practices.append("WAIT TIME")
                        
                        if instance_key not in instance_utterances:
                            instance_utterances[instance_key] = {
                                'utterances': [],
                                'practices': practices
                            }
                        else:
                            # Merge practices if instance already exists
                            instance_utterances[instance_key]['practices'] = list(set(instance_utterances[instance_key]['practices'] + practices))
                        instance_utterances[instance_key]['utterances'].append(utterance)
                
                if prompt_type == "praise" or prompt_type == "all":
                    # Check for PRAISE umbrella instance
                    if row.get('praises') == 1 and row.get('praises_instance_id', 0) > 0:
                        instance_id = int(row.get('praises_instance_id', 0))
                        instance_key = f"INSTANCE #{instance_id}"
                        
                        # Collect all praise labels for this instance
                        practices = []
                        if row.get('praises') == 1:
                            practices.append("PRAISE")
                        if row.get('academic_specific_praises') == 1:
                            practices.append("ACADEMIC PRAISE")
                        if row.get('behavior_specific_praises') == 1:
                            practices.append("BEHAVIOR PRAISE")
                        
                        if instance_key not in instance_utterances:
                            instance_utterances[instance_key] = {
                                'utterances': [],
                                'practices': practices
                            }
                        else:
                            # Merge practices if instance already exists
                            instance_utterances[instance_key]['practices'] = list(set(instance_utterances[instance_key]['practices'] + practices))
                        instance_utterances[instance_key]['utterances'].append(utterance)
            
            # Create instance annotations from collected utterances
            for instance_key, instance_data in instance_utterances.items():
                # Filter out empty/untranscribed utterances for instance annotations
                filtered_utterances = []
                for utt in instance_data['utterances']:
                    if (not pd.isna(utt) and utt != '' and utt != '[Untranscribed]' 
                        and not utt.startswith('[') and not utt.endswith(']')):
                        filtered_utterances.append(utt)
                
                if filtered_utterances:
                    combined_utterance = " ".join(filtered_utterances)
                    
                    # Define order for practice labels
                    otr_order = ["OTR", "OEQ", "PRESS", "WAIT TIME"]
                    praise_order = ["PRAISE", "ACADEMIC PRAISE", "BEHAVIOR PRAISE"]
                    
                    # Sort practices according to predefined order
                    practices = instance_data['practices']
                    if any(p in otr_order for p in practices):
                        # Use OTR order for OTR pipeline practices
                        ordered_practices = [p for p in otr_order if p in practices]
                    else:
                        # Use praise order for praise pipeline practices
                        ordered_practices = [p for p in praise_order if p in practices]
                    
                    practices_str = ", ".join(ordered_practices)
                    bubble_instances.append(f"{instance_key} [{practices_str}]: \"{combined_utterance}\"")
            
            # Create the combined utterance for this bubble
            if utterances:
                combined_utterance = " ".join(utterances)
                formatted_lines.append(f"{speaker.upper()}: {combined_utterance}")
                
                # Add instance annotations if any
                for instance in bubble_instances:
                    formatted_lines.append(f"  - {instance}")
            elif bubble_instances:
                # If there are instances but no utterances, still show the speaker
                # Get duration from the first row in the group
                duration = group.iloc[0].get('duration_in_seconds', 0)
                formatted_lines.append(f"{speaker.upper()}: [{speaker} talk for {duration:.1f} seconds]")
                
                # Add instance annotations
                for instance in bubble_instances:
                    formatted_lines.append(f"  - {instance}")
            elif speaker == 'silence':
                # Handle silence bubbles with no utterances and no teaching practices
                duration = group.iloc[0].get('duration_in_seconds', 0)
                formatted_lines.append(f"{speaker.upper()}: [silence for {duration:.1f} seconds]")
            else:
                # Handle other speakers with no utterances and no teaching practices
                duration = group.iloc[0].get('duration_in_seconds', 0)
                formatted_lines.append(f"{speaker.upper()}: [{speaker} talk for {duration:.1f} seconds]")
        
        return '\n'.join(formatted_lines)
        
    except Exception as e:
        return f"Error reading transcript: {str(e)}"

def save_model_output(report_id: str, prompt_name: str, model_output: Dict[Any, Any]) -> str:
    """
    Save model output as JSON file in the results/{ID}/ folder.
    
    Args:
        report_id: The report/transcript ID (e.g., "137984")
        prompt_name: Name of the prompt used (e.g., "praise", "otr")
        model_output: The JSON output from the model
        
    Returns:
        Path to the saved file
    """
    try:
        # Create results directory if it doesn't exist
        results_dir = f"results/{report_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename based on prompt name
        filename = f"{prompt_name}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Save the model output as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_output, f, indent=2, ensure_ascii=False)
        
        return filepath
        
    except Exception as e:
        return f"Error saving model output: {str(e)}"

def get_available_prompts() -> List[str]:
    """
    Get list of available prompt names from the prompts/system/ directory.
    
    Returns:
        List of prompt names (without .txt extension)
    """
    try:
        # Use absolute path to ensure we find the prompts directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(base_dir, "prompts", "system")
        
        if not os.path.exists(prompts_dir):
            return []
        
        prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
        return [f.replace('.txt', '') for f in prompt_files]
        
    except Exception as e:
        print(f"Error reading prompts directory: {str(e)}")
        return []

def get_all_report_ids() -> List[str]:
    """
    Scan the transcripts directory to find all report IDs.
    
    Returns:
        List of report IDs (CSV filenames without .csv extension)
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        transcripts_dir = os.path.join(base_dir, "transcripts")
        
        if not os.path.exists(transcripts_dir):
            return []
        
        # Get all CSV files in transcripts/ directory
        report_ids = []
        for item in os.listdir(transcripts_dir):
            if item.endswith('.csv'):
                report_id = item.replace('.csv', '')
                report_ids.append(report_id)
        
        return sorted(report_ids)
        
    except Exception as e:
        print(f"Error scanning transcripts directory: {str(e)}")
        return []

def load_prompt(prompt_name: str, prompt_type: str = "system") -> str:
    """
    Load a prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt (without .txt extension)
        prompt_type: Type of prompt ("system" or "user")
        
    Returns:
        The prompt content as a string
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "prompts", prompt_type, f"{prompt_name}.txt")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading prompt: {str(e)}"

def create_user_prompt(transcript: str, prompt_name: str) -> str:
    """
    Create a user prompt by loading the user prompt file and replacing the transcript placeholder.
    
    Args:
        transcript: The formatted transcript to include
        prompt_name: Name of the prompt (e.g., "otr", "praise")
        
    Returns:
        User prompt with transcript included
    """
    user_prompt_template = load_prompt(prompt_name, "user")
    return user_prompt_template.format(transcript=transcript)

def process_transcript_with_prompt(csv_file_path: str, prompt_name: str, report_id: str, model: str = "gpt-5", overwrite: bool = False) -> Dict[str, Any]:
    """
    Process a transcript with a specific prompt using OpenAI API and save the results.
    
    Args:
        csv_file_path: Path to the CSV transcript file
        prompt_name: Name of the prompt to use (e.g., "praise", "otr")
        report_id: The report/transcript ID
        model: OpenAI model to use (default: "gpt-5")
        overwrite: Whether to overwrite existing result files (default: False)
        
    Returns:
        Dictionary with processing results and file paths
    """
    try:
        # Check if result file already exists
        results_dir = f"results/{report_id}"
        result_file = os.path.join(results_dir, f"{prompt_name}.json")
        
        if not overwrite and os.path.exists(result_file):
            return {
                "success": True,
                "prompt_name": prompt_name,
                "saved_path": result_file,
                "skipped": True,
                "message": f"Result file already exists: {result_file}. Use --overwrite to regenerate."
            }
        
        # Load the system prompt
        system_prompt = load_prompt(prompt_name, "system")
        
        # Format the transcript based on prompt type
        prompt_type = "otr" if prompt_name == "otr" else "praise" if prompt_name == "praise" else "all"
        formatted_transcript = format_transcript(csv_file_path, prompt_type)
        
        # Create user prompt with transcript
        user_prompt = create_user_prompt(formatted_transcript, prompt_name)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract the response content
        model_response = response.choices[0].message.content
        
        # Extract usage information for cost tracking
        usage = response.usage
        cost_info = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None
        }
        
        # Try to parse JSON response
        try:
            model_output = json.loads(model_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, wrap the response
            model_output = {
                "prompt_used": prompt_name,
                "transcript_id": report_id,
                "processing_timestamp": pd.Timestamp.now().isoformat(),
                "raw_response": model_response,
                "error": "Failed to parse JSON response"
            }
        
        # Add metadata
        model_output.update({
            "prompt_used": prompt_name,
            "transcript_id": report_id,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
            "model_used": model,
            "transcript_length": len(formatted_transcript),
            "api_usage": cost_info
        })
        
        # Save the model output
        saved_path = save_model_output(report_id, prompt_name, model_output)
        
        return {
            "success": True,
            "prompt_name": prompt_name,
            "saved_path": saved_path,
            "model_output": model_output,
            "model_used": model
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "prompt_name": prompt_name
        }

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process classroom transcripts with teaching practice prompts")
    parser.add_argument("--model", default="gpt-5",
                       help="OpenAI model to use (default: gpt-5)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing result files")
    parser.add_argument("--show-transcript", action="store_true",
                       help="Show formatted transcript output")
    parser.add_argument("--report_id", 
                       help="Process only the specified report ID (e.g., '136547'). If not provided, processes all available reports.")
    
    args = parser.parse_args()
    
    # Get reports to process
    if args.report_id:
        # Process only the specified report
        if not os.path.exists(f"transcripts/{args.report_id}.csv"):
            print(f"ERROR: Transcript file not found: transcripts/{args.report_id}.csv")
            exit(1)
        report_ids = [args.report_id]
        print(f"Processing specific report: {args.report_id}")
    else:
        # Process all available reports
        report_ids = get_all_report_ids()
        if not report_ids:
            print("ERROR: No reports found in transcripts/ directory")
            exit(1)
        print(f"Found {len(report_ids)} reports: {', '.join(report_ids)}")
    
    # Get available prompts
    available_prompts = get_available_prompts()
    if not available_prompts:
        print("ERROR: No prompts found in prompts/system/ directory")
        exit(1)
    
    print(f"Available prompts: {', '.join(available_prompts)}")
    
    # Process each report
    for report_id in report_ids:
        print(f"\n{'='*60}")
        print(f"PROCESSING REPORT: {report_id}")
        print(f"{'='*60}")
        
        csv_path = f"transcripts/{report_id}.csv"
        
        if not os.path.exists(csv_path):
            print(f"ERROR: Transcript file not found: {csv_path}")
            continue
        
        if args.show_transcript:
            print("\n\n=== TRANSCRIPT ===")
            print(format_transcript(csv_path))
        
        print(f"\n--- Processing {report_id} with all prompts ---")
        
        # Process transcript with each available prompt
        for prompt_name in available_prompts:
            print(f"\n--- Processing {prompt_name} prompt for {report_id} ---")
            result = process_transcript_with_prompt(csv_path, prompt_name, report_id, model=args.model, overwrite=args.overwrite)
            
            if result["success"]:
                if result.get("skipped"):
                    print(f"SKIPPED {prompt_name}: {result['message']}")
                else:
                    print(f"SUCCESS: Processed {prompt_name} for {report_id}")
                    print(f"Saved to: {result['saved_path']}")
                    print(f"Model used: {result.get('model_used', 'Unknown')}")
            else:
                print(f"ERROR processing {prompt_name} for {report_id}: {result['error']}")
        
        print(f"\nResults for {report_id} saved in: results/{report_id}/")
    
    print(f"\nProcessing complete for all {len(report_ids)} reports!")
