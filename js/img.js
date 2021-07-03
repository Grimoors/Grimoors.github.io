function change_image() {
    var x = document.getElementById('changingimage')
    var str = document.getElementById('changingimage').src;
    if (str.includes("1.jpeg")) {
        x.src = "../img/2.jpeg";
    } else if (str.includes("2.jpeg")) {
        x.src = "../img/3.jpeg";
    } else if (str.includes("3.jpeg")) {
        x.src = "../img/4.jpeg";
    } else if (str.includes("4.jpeg")) {
        x.src = "../img/5.jpeg";
    } else if (str.includes("5.jpeg")) {
        x.src = "../img/1.jpeg";
    } else {
        x.src = "";
    }
}