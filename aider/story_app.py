import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from crewai_tools.tools import FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_vertexai import ChatVertexAI, VertexAI
import os
import uuid
import subprocess

# Initialize Vertex AI
PROJECT_ID = 'my-project-0004-346516'
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize models
llm = ChatVertexAI(
    model_name='gemini-1.5-pro',
    project_id=PROJECT_ID,
    location=LOCATION,
)

model = VertexAI(model_name="gemini-1.5-pro")

def generate_pro(input_prompt):
    model = GenerativeModel("gemini-1.5-pro")
    full_prompt = '''summarize the prompt below and do note prompt below will be send to imagen model to create images  so please clean up any sensitve words and replace them into unblocked words like replace girl or lady can be replaced by female human and remove any names to make the prompt simple and easy  : ''' + input_prompt
    responses = model.generate_content(
        input_prompt,
        generation_config={
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 1
        },
        stream=False,
    )
    return responses.text

@tool
def generateimage(chapter_content_and_character_details: str) -> str:
    """
    Generates an image using Imagen for a given chapter content and character details and return the image_url.
    """
    image_generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

    chapter_content_and_character_details_refined = generate_pro(chapter_content_and_character_details)
    
    prompt = f"Image is about: {chapter_content_and_character_details_refined}. Style: Illustration. Create an illustration incorporating a vivid palette with an emphasis on shades of azure and emerald, augmented by splashes of gold for contrast and visual interest. The style should evoke the intricate detail and whimsy of early 20th-century storybook illustrations, blending realism with fantastical elements to create a sense of wonder and enchantment. The composition should be rich in texture, with a soft, luminous lighting that enhances the magical atmosphere. Attention to the interplay of light and shadow will add depth and dimensionality, inviting the viewer to delve into the scene. DON'T include ANY text in this image. DON'T include colour palettes in this image."

    response = image_generation_model.generate_images(
        prompt=prompt,
        aspect_ratio="1:1",
        number_of_images=1,
        safety_filter_level="block_few",
        person_generation="allow_adult",
    )
    
    # image_url = response.images[0].url
    image_url_name = str(uuid.uuid4())
    
    filepath = os.path.join(os.getcwd(), "images1/")

    image_url = filepath+image_url_name+".jpg"
    response.images[0].save(location=image_url)
    print("url",image_url)

    return(image_url)

@tool
def convermarkdowntopdf(markdownfile_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using mdpdf.
    """
    output_file = os.path.splitext(markdownfile_name)[0] + '.pdf'
    cmd = ['mdpdf', '--output', output_file, markdownfile_name]
    subprocess.run(cmd, check=True)
    return output_file

def create_story(topic):
    # Create agents
    story_outliner = Agent(
        role='Story Outliner',
        goal=f'Develop an outline for a children\'s storybook about {topic}, including chapter titles and characters for 5 chapters.',
        backstory="An imaginative creator who lays the foundation of captivating stories for children.",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    story_writer = Agent(
        role='Story Writer',
        goal='Write the full content of the story for all 5 chapters, each chapter 100 words, weaving together the narratives and characters outlined.',
        backstory="A talented storyteller who brings to life the world and characters outlined, crafting engaging and imaginative tales for children.",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    image_generator = Agent(
        role='Image Generator',
        goal='Generate one image per chapter content and provide a description of what was generated. Include both the image URL and a narrative description of the scene depicted.',
        backstory="A creative AI specialized in visual storytelling, bringing each chapter to life through imaginative imagery while providing detailed descriptions of the generated scenes.",
        verbose=True,
        llm=llm,
        tools=[generateimage],
        allow_delegation=False
    )

    # Create tasks
    task_outline = Task(
        description=f'Create an outline for the children\'s storybook about {topic}, detailing chapter titles and character descriptions for 5 chapters.',
        agent=story_outliner,
        expected_output='A structured outline document containing 5 chapter titles, with detailed character descriptions and the main plot points for each chapter.'
    )

    task_write = Task(
        description='Using the outline provided, write the full story content for all chapters, ensuring a cohesive and engaging narrative for children.',
        agent=story_writer,
        expected_output='A complete manuscript of the children\'s storybook with 5 chapters. Each chapter should follow the provided outline and integrate the characters and plot points into a cohesive narrative.'
    )

    task_image_generate = Task(
        description='Generate 5 images that capture the essence of the story, one for each chapter. For each image, provide a description of the scene depicted.',
        agent=image_generator,
        expected_output='A set of 5 digital image files with corresponding descriptions that visually represent each chapter of the storybook. Each output should include both the image file path and a narrative description of what was generated, incorporating elements from the characters and plot as described in the outline.'
    )

    # Create and run crew
    crew = Crew(
        agents=[story_outliner, story_writer, image_generator],
        tasks=[task_outline, task_write, task_image_generate],
        verbose=True,
        process=Process.sequential
    )

    return crew.kickoff()

def main():
    st.title("Children's Story Generator")
    st.write("Generate an illustrated children's story about any topic!")

    # User input
    topic = st.text_input("Enter a topic for your story:", "Animals")

    if st.button("Generate Story"):
        with st.spinner("Creating your story... This may take a few minutes."):
            try:
                result = create_story(topic)
                st.success("Story generated successfully!")
                st.write(result)
                
                # Display generated images with their stories
                image_dir = os.path.join(os.getcwd(), "images1")
                if os.path.exists(image_dir):
                    for image_file in os.listdir(image_dir):
                        if image_file.endswith(".jpg"):
                            image_path = os.path.join(image_dir, image_file)
                            st.image(image_path)
                            # Extract chapter number from result and display corresponding story
                            chapter_match = re.search(r"Chapter (\d+).*?:(.*?)(?=Chapter \d+|$)", result, re.DOTALL)
                            if chapter_match:
                                st.markdown(f"### Chapter {chapter_match.group(1)}")
                                st.markdown(chapter_match.group(2).strip())
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
