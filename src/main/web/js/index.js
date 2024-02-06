const swiper = new Swiper(".home_swiper", {
  loop: true,
  slidesPerView: 1,
  auto: true,
  autoplay: {
    delay: 2500,
    disableOnInteraction: false,
  },
});

var navbar = document.querySelector(".nav");

const stickynav = () => {
  if (document.documentElement.scrollTop > 0) {
    navbar.classList.add("active");
    console.log("scrolled");
  } else {
    navbar.classList.remove("active");
  }
};

document.addEventListener("scroll", stickynav);
stickynav();
