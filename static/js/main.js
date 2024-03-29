const imageWidth = 550;

document.addEventListener("DOMContentLoaded", function () {
  const picture_input = document.getElementById("picture_input");
  const picture_output = document.getElementById("picture_output");
  const slider = document.getElementById("slider");
  const number = document.getElementById("width_text");

  let slider_range = 1;

  function sendImageToBackend(file, route) {
    const blob = URL.createObjectURL(file);

    let imagePacket = new FormData();
    imagePacket.append("picture", file);
    imagePacket.append("scaling_factor", slider.value);

    try {
      fetch(route, {
        method: "POST",
        body: imagePacket,
      })
        .then((response) => response.json())
        .then((data) => {
          // Don't cache the images
          URL.revokeObjectURL(blob);
          picture_output.src = "data:image/png;base64," + data.image;
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } catch (error) {
      alert(error);
    }
  }

  picture_input.addEventListener("change", function (e) {
    picture_output.src = "";

    const file = e.target.files[0];

    if (file) {
      const image = new Image();
      image.src = URL.createObjectURL(file);

      image.onload = () => {
        const width = image.naturalWidth;
        const height = image.naturalHeight;

        origWidth = width;

        slider_range = Math.min(imageWidth, width);

        slider.min = 20;
        slider.max = slider_range;
        slider.value = slider_range;
        slider.style = `width: ${slider_range}px`;

        number.innerHTML = `Width: 100%`;
        slider.hidden = false;
        number.hidden = false;
        sendImageToBackend(picture_input.files[0], "/receive_image");
      };
    }
  });

  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  const getModifiedImage = debounce(function (route) {
    picture_output.classList.add("image-scaling");

    let dataPacket = new FormData();
    dataPacket.append("scaling_factor", slider.value);

    fetch(route, {
      method: "POST",
      body: dataPacket,
    })
      .then((response) => response.json())
      .then((data) => {
        picture_output.style.transform = "none";
        picture_output.src = "data:image/png;base64," + data.image;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }, 100);

  slider.oninput = function () {
    number.innerHTML = `Width: ${Math.round(
      (this.value / slider_range) * 100
    )}%`;
    getModifiedImage("/process_image");
  };
});
