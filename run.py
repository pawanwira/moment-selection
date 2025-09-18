import csv
import pandas as pd
import json
import os
import argparse
from typing import List, Dict, Any
import openai
from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize client only when needed
client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client


# Pydantic Models for Structured Output
class ExemplarMoment(BaseModel):
    instance_id: str
    quote: str
    post_quote_description: str


class ExemplarMomentsWithRationale(BaseModel):
    exemplars: List[ExemplarMoment] = Field(min_items=0, max_items=3)
    rationale: str


class OTRMoments(BaseModel):
    otr: ExemplarMomentsWithRationale
    oeq: ExemplarMomentsWithRationale
    press: ExemplarMomentsWithRationale
    wait_time: ExemplarMomentsWithRationale


class PraiseMoments(BaseModel):
    praise: ExemplarMomentsWithRationale
    academic_specific_praise: ExemplarMomentsWithRationale
    behavior_specific_praise: ExemplarMomentsWithRationale


def get_openai_LLM_response(generate_prompt_fn, args, LLM_model_name, temperature, response_format = None, stop_sequences = None):
    """
    Given a function that generates the prompt to the LLM (generate_prompt_fn), a list of arguments for that function (args), the LLM model name, the temperature value, the LLM response format and the stop sequences,
    this function pings the Open AI (compatible with openai version 1.52.2) LLM and returns the response.
    """
    from airflow.exceptions import AirflowException
    from airflow.models.variable import Variable
    from openai import OpenAI

    # set timeout per openai call.
    client = OpenAI(api_key=Variable.get("OPEN_AI_API_KEY"), timeout=float(Variable.get("OPEN_AI_CALL_TIMEOUT", 120.0)))
    
    kwargs = {
        "messages": generate_prompt_fn(*args),
        "model": LLM_model_name,
        "temperature": temperature
        }

    if response_format is not None:
        kwargs["response_format"] = response_format
    if stop_sequences is not None:
        kwargs["stop"] = stop_sequences

    # try 2 times in case a call fails
    num_tries = 2
    num_tries = max(1, num_tries)
    exceptions = []
    for i in range(num_tries):
        if i > 0:
            print(f"Retry {i}...")
        try:
            response = get_client().beta.chat.completions.parse(**kwargs)
            break
        except Exception as e:
            print(f"OpenAI exception: {e}")
            exceptions.append(e)

    if len(exceptions) == num_tries:
        raise AirflowException(exceptions[-1])
    
    return response


def generate_llm_prompt(system_prompt, user_prompt, assistant_prompt=None):
    """Generate LLM prompt following Airflow pattern."""
    prompts = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user", 
        "content": user_prompt
    }]
    if assistant_prompt:
        prompts.append({
            "role": "assistant",
            "content": assistant_prompt
        })
    return prompts


def select_exemplary_otr_moments(transcript: str, model: str = "gpt-5", temperature: float = 0.1) -> OTRMoments:
    """Select exemplary OTR moments using structured output."""
    system_prompt = """You are an expert educator reviewing a classroom transcript. The transcript is labeled with the following teaching practices: Giving students opportunities to respond, Asking open-ended questions, Pressing students to explain, and Waiting after asking a question. Your goal is to choose the top 3 exemplar moments for each of the 4 teaching practices to surface to teachers in a lesson report.

Transcript format:
- Each speaker's turn is shown as SPEAKER: utterance
- Teaching practice instance annotations appear below relevant turns as:   - INSTANCE #{instance id} [practice label(s)]: "utterance within turn"
- [speaker talk for X seconds] indicates untranscribed speech
- When selecting exemplar moments for a specific practice, consider all instances that contain that practice label in their brackets

Teaching practice definitions:

Give students opportunities to respond (OTR):
When teacher asks students to verbally respond to a prompt or question.
Frequently giving your students opportunities to respond leads to more engaged students and fewer disruptions. It also gives you more chances to check for understanding.

Ask open-ended questions (OEQ):
Open-ended questions have many acceptable responses, unlike closed-ended questions which have a single correct answer.
Open-ended questions create more opportunities for students to talk, express thoughts in detail,  think critically, and engage in richer classroom discussions.

Press students to explain (PRESS):
When teacher asks a student to explain their thought process.
This highly effective practice helps students develop deeper conceptual understanding and better communication skills. It also helps you better assess their understanding and point of view.

Wait after asking a question (WAIT TIME):
Waiting at least 2.5 seconds after asking students to respond to a question.
This highly effective practice gives students time to organize their thoughts. It also gives a diversity of students the chance to respond, rather than just the first student to raise their hand.

Instructions:
1. Review all instances of each teaching practice (OTR, OEQ, PRESS, WAIT TIME) with context.
2. For each teaching practice, carefully select 3 moments that best exemplify that teaching practice (based on the definitions above), optimizing for the following dimensions:
    - Pedagogical impact: Does the moment demonstrate clear educational value and impact on students?
    - Diversity & range: Do the 3 examples show different contexts, approaches, linguistic variety, and interactions with students?
    - Relevance to learning: How well does it connect to the lesson content and student learning?
    - Substance & depth: Does it provide meaningful content for teacher reflection (avoiding overly brief examples)?
    - Linguistic richness: Ideally, the selected utterance should be linguistically rich and demonstrate thoughtful use of language
    - (OEQ only): Does it involve high cognitive demand? A high cognitive level question asks students to do any of the following: explain, compare, assess, classify, interpret, infer, hypothesize, or synthesize. A high cognitive level question is NOT one which merely asks students for a report or recitation of facts, it is NOT a question that asks students to merely recall, recite, recognize, match, quote, report, label, list, memorize or identify something.
    Note: If there are less than 3 instances of a teaching practice, select all available instances.
3. For each selected moment, provide:
    - Quote: the exact teaching practice instance quote.
    - Instance ID: The number after the # symbol corresponding to that teaching practice you're selecting the exemplar for (e.g., from "OTR #1" the instance ID is "1", from "OEQ #2" the instance ID is "2").
    - Post-quote description (1 complete sentence) explaining how it connects to lesson/student learning. Note: This description will be surfaced to the teacher whose recording is being analyzed to provide them with more context about the moment highlighted.
4. For each teaching practice type, provide a rationale explaining why the 3 moments, taken together, give a strong, diverse picture of that teaching practice. Base your rationale on the selection dimensions listed above.

You must respond with a valid JSON object that exactly matches the required schema. Do not include any text outside the JSON structure. Each "exemplars" array must contain exactly 3 items (or fewer if there are fewer than 3 instances available)."""

    user_prompt = f"Analyze this classroom transcript and select exemplar moments for each teaching practice:\n\n{transcript}"
    assistant_prompt = None
    
    response = get_openai_LLM_response(
        generate_prompt_fn=generate_llm_prompt,
        args=[system_prompt, user_prompt, assistant_prompt],
        LLM_model_name=model,
        temperature=temperature,
        response_format=OTRMoments
    )
    
    return response.choices[0].message.parsed


def select_exemplary_praise_moments(transcript: str, model: str = "gpt-5", temperature: float = 0.1) -> PraiseMoments:
    """Select exemplary praise moments using structured output."""
    system_prompt = """You are an expert educator reviewing a classroom transcript. The transcript is labeled with praise practices: Praise students, Give academic-specific praise, and Give behavior-specific praise. Your task is to select the top 3 exemplar moments for each of the 3 praise categories to surface to teachers in a lesson report.

Transcript format:
- Each speaker's turn is shown as SPEAKER: utterance
- Praise instance annotations appear below relevant turns as:   - INSTANCE #{instance id} [practice label(s)]: "utterance within turn"
- [speaker talk for X seconds] indicates untranscribed speech
- When selecting exemplar moments for a specific practice, consider all instances that contain that practice label in their brackets

Praise definitions:

Praise students (PRAISE):
Positively acknowledging students with both general and specific praise.
Praise creates a positive classroom culture and encourages students to repeat expected behaviors.

Give academic-specific praise (ACADEMIC PRAISE):
Positively acknowledging academic thinking, effort, process, or contribution.
This highly effective practice helps create a positive classroom culture, encourages participation, and promotes effective academic strategies.

Give behavior-specific praise (BEHAVIOR PRAISE):
Positively acknowledging or narrating behaviors that align with classroom norms.
This highly effective practice helps create a positive classroom culture and encourages students to repeat expected behaviors.

Instructions:
1. Review all instances of each praise category (PRAISE, ACADEMIC PRAISE, BEHAVIOR PRAISE) with context.
2. For each praise category, carefully select 3 moments that best exemplify that category (based on the definitions above), optimizing for the following dimensions:
    - Pedagogical impact: Does the moment demonstrate clear educational value and impact on students?
    - Diversity & range: Do the 3 examples show different contexts, approaches, linguistic variety, and interactions with students?
    - Substance & depth: Does it provide meaningful content for teacher reflection (avoiding overly brief examples)?
    - Linguistic richness: Ideally, the selected utterance should be linguistically rich and demonstrate thoughtful use of language
    - Specificity: Does it demonstrate high specificity in targeting academic effort or behavior?
    Note: If there are less than 3 instances of a praise category, select all available instances.
3. For each selected moment, provide:
    - Quote: the exact praise instance quote.
    - Instance ID: The number after the # symbol corresponding to that teaching practice you're selecting the exemplar for (e.g., from "PRAISE #1" the instance ID is "1", from "ACADEMIC PRAISE #2" the instance ID is "2").
    - Post-quote description (1 complete sentence) explaining how it supports student learning/engagement. Note: This description will be surfaced to the teacher whose recording is being analyzed to provide them with more context about the moment highlighted.
4. For each praise category, provide a rationale explaining why the 3 moments, taken together, give a strong, diverse picture of that praise category. Base your rationale on the selection dimensions listed above.

You must respond with a valid JSON object that exactly matches the required schema. Do not include any text outside the JSON structure. Each "exemplars" array must contain exactly 3 items (or fewer if there are fewer than 3 instances available)."""

    user_prompt = f"Analyze this classroom transcript and select exemplar moments for each praise category:\n\n{transcript}"
    assistant_prompt = None
    
    response = get_openai_LLM_response(
        generate_prompt_fn=generate_llm_prompt,
        args=[system_prompt, user_prompt, assistant_prompt],
        LLM_model_name=model,
        temperature=temperature,
        response_format=PraiseMoments
    )
    
    return response.choices[0].message.parsed


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





def process_transcript(csv_file_path: str, prompt_name: str, report_id: str, model: str = "gpt-5", overwrite: bool = False) -> Dict[str, Any]:
    """
    Process a transcript using structured output with Pydantic models.
    
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
        
        # Format the transcript based on prompt type
        prompt_type = "otr" if prompt_name == "otr" else "praise" if prompt_name == "praise" else "all"
        formatted_transcript = format_transcript(csv_file_path, prompt_type)
        
        # Analyze using structured output
        if prompt_name == "otr":
            analysis_result = select_exemplary_otr_moments(formatted_transcript, model=model)
        elif prompt_name == "praise":
            analysis_result = select_exemplary_praise_moments(formatted_transcript, model=model)
        else:
            raise ValueError(f"Unknown prompt name: {prompt_name}")
        
        # Convert to dictionary for JSON serialization
        model_output = analysis_result.model_dump()
        
        # Add metadata
        model_output.update({
            "prompt_used": prompt_name,
            "transcript_id": report_id,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
            "model_used": model,
            "transcript_length": len(formatted_transcript),
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
    
    # Process OTR and praise prompts
    available_prompts = ["otr", "praise"]
    print(f"Processing prompts: {', '.join(available_prompts)}")
    
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
            result = process_transcript(csv_path, prompt_name, report_id, model=args.model, overwrite=args.overwrite)
            
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
