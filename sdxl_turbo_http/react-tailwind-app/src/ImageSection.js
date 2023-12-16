import React from 'react';

function ImageSection() {
    const [state, setState] = React.useState({
        prompt: "",
        running: false,
        image: null,
    });
    const { image, prompt, running } = state

  const update = (e) => {
      const {name, value} = e.target;
      setState(prev => ({...prev,
          prompt: value,
            running: true,
      }))

      fetch(`/process/?prompt=${value}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.blob();
      })
      .then((data) => {
          setState(prev => ({...prev,
              image: data, running: false
          }))
      })
      .catch((error) => {
      });
  }
  
    const imageUrl = (image)? URL.createObjectURL(image): null

  return (
    <div className="container mx-auto py-8">
      <div className="max-w-xl mx-auto bg-white rounded-lg shadow-md overflow-hidden">
        <img
          src={imageUrl}
          alt="Placeholder Image"
          className="w-full h-full object-cover"
        />
        <div className="p-4">
          <h1 className="text-xl font-semibold">Image Area</h1>
          <p className="mt-2">Your image description goes here.</p>
            <textarea
                name="prompt"
                value={prompt}
                onChange={update}
                className="w-full h-24 px-3 py-2 text-gray-700 border rounded-lg focus:outline-none"
                placeholder="Enter your prompt here..."/>
          <div className="mt-4 flex justify-between">
            <button className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded"
                onClick={() => update( { target: {name: "", value: prompt}})}>
                More
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ImageSection;

