document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  form.addEventListener("submit", function () {
    // Show loading spinner
    const btn = form.querySelector("button");
    btn.innerText = "Loading...";
    btn.disabled = true;
  });
});
