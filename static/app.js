const form = document.getElementById("upload-form");
const submitButton = document.getElementById("submit-button");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const artistNode = document.getElementById("artist-name");
const titleNode = document.getElementById("song-title");
const galleryNode = document.getElementById("gallery");

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function clearResults() {
  artistNode.textContent = "";
  titleNode.textContent = "";
  galleryNode.innerHTML = "";
  resultBox.classList.add("hidden");
}

function renderPhotos(photos) {
  galleryNode.innerHTML = "";
  if (!photos.length) {
    const empty = document.createElement("p");
    empty.textContent = "Фотографии исполнителя не найдены.";
    galleryNode.appendChild(empty);
    return;
  }

  photos.slice(0, 5).forEach((url, index) => {
    const card = document.createElement("div");
    card.className = "photo-card";

    const image = document.createElement("img");
    image.src = url;
    image.alt = `Фото исполнителя #${index + 1}`;
    image.loading = "lazy";

    card.appendChild(image);
    galleryNode.appendChild(card);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearResults();
  setStatus("Обрабатываем MP3...");

  submitButton.disabled = true;

  const formData = new FormData(form);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Ошибка обработки файла.");
    }

    artistNode.textContent = payload.artist;
    titleNode.textContent = payload.title;
    renderPhotos(payload.photos || []);
    resultBox.classList.remove("hidden");
    setStatus("Готово. Исполнитель и песня распознаны.");
  } catch (error) {
    setStatus(error.message || "Не удалось выполнить запрос.", true);
  } finally {
    submitButton.disabled = false;
  }
});
