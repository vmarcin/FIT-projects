/*
Project: Rollager
File:    script.js
Authors: Marcin Vladimir (xmarci10@stud.fit.vutbr.cz)
         Liscinsky Matus (xlisci02@stud.fit.vutbr.cz)
*/

/**************************************************************************/
/*******************          Global variables          *******************/
/**************************************************************************/
var reader = new FileReader(),                  // file reader
    img1 = new Image(),                         // first image object
    img2 = new Image(),                         // second image object
    phase = 0;                                  // current phase
/**************************************************************************/
/*******************       end of Global variables      *******************/



/**************************************************************************/
/*******************          Helper functions          *******************/
/**************************************************************************/

// Function that gets the element from document by its id
function e(id) {
    return document.getElementById(id);
}
// This function returns random number from interval min, max
function getRandom(min, max) {
    return Math.random() * (max - min) + min;
}
/**************************************************************************/
/*******************       end of Helper functions      *******************/


/**************************************************************************/
/****************    Website controlling functions      *******************/
/**************************************************************************/

// Initialization function, called on page load
function init() {
    e("shape-width").value = 25;                // init width of clipping shape
    e("shape-height").value = 25;               // init height of clipping shape
    $("#download").click(download);             // setting download function on 'Download' button
}

// Function that hides the menu, to make a user see the whole picture
function mouseDown() {
    // menu panel
    e("settings").style.visibility = "hidden";
    // buttons
    e("input2button").style.visibility = "hidden";
    e("generate").style.visibility = "hidden";
    e("download").style.visibility = "hidden";
}

// Function that shows the menu back, after releasing the mouse button
function mouseUp() {
    // menu panel
    e("settings").style.visibility = "visible";
    // buttons are showed according to what phase were hidden
    if (phase > 0) e("input2button").style.visibility = "visible";
    if (phase > 1) e("generate").style.visibility = "visible";
    if (phase > 2) e("download").style.visibility = "visible";
}

// Wrapper function that calls the corresponding drawing function
function draw() {
    phase = 3; // drawing phase
    if ($("#radioCircles").is(":checked")) {
        drawCircle(true);
    } else if ($("#radioElipses").is(":checked")) {
        drawCircle(false);
    } else {
        drawTriangles();
    }
}

// Function that changes the settings text, when user wants to work with circles
$("#radioCircles").click(function () {
    e("shape-width-span").innerText = "Min Radius:";     // min circle radius
    e("shape-height-span").innerText = "Max Radius:";    // max circle radius
});


// Function that changes the settings text, when user wants to work with ellipses
$("#radioElipses").click(function () {
    e("shape-width-span").innerText = "Min Radius:";     // min ellipse radius
    e("shape-height-span").innerText = "Max Radius:";    // max ellipse radius
});

// Function that changes the settings text, when user wants to work with triangles
$("#radioTriangles").click(function () {
    e("shape-width-span").innerText = "Shape Width:";    // triangle width
    e("shape-height-span").innerText = "Shape Height:";  // triangle height
});

$("#outline").click(function () {
    // this function will get executed every time the #outline element is clicked
    if ($(this).is(":checked")) {
        // "this" refers to the element that fired the event
        $("#stroke-width").prop("disabled", false);
    }
    else {
        $("#stroke-width").prop("disabled", true);
    }
});

// This function reads the first image's data and save them to the img1 image object
function uploadImage1() {
    var input = e("input1");       // input element

    input.onchange = function () {
        reader.readAsDataURL(input.files[0]);            // read image data
        $("#input2button").css("visibility", "visible"); // make the next button visible
        phase = 1;                                       // change the phase
        reader.onload = function () {                    // set image source
            img1.src = reader.result;
        };
    };
}

// This function reads the second image's data and save them to the img2 image object
function uploadImage2() {
    var input = e("input2");       // input element

    input.onchange = function () {
        reader.readAsDataURL(input.files[0]);            // read image data
        $("#generate").css("visibility", "visible");     // set the generate button visible
        phase = 2;                                       // change the phase
        reader.onload = function () {                    // set image source
            img2.src = reader.result;
        };
    };
}

// Download function
function download() {
    // console.log("download");
    // Get the image data
    var fileData = e("rendering").toDataURL("image/png");
    fileData = fileData.replace(
        /^data:image\/png/,
        "data:application/octet-stream"
    );
    // Set the link to download data
    e("download").href = fileData;
}
/*****************************************************************************/
/******************* end of Website controlling functions  *******************/

/**************************************************************************/
/****************   Drawing and calcutation functions   *******************/
/**************************************************************************/

// Main drawing function for rollage using circles/ellipses
function drawCircle(isCircle) {
    // Get canvas object and its context
    var canvas = e("rendering");
    var ctx = canvas.getContext("2d");
    var strokeColor = e("colorpicker").value;

    // User's window size
    var H = window.screen.height;
    var W = window.screen.width;

    // Get minimum and maximum value for circle/ellipse radius from user
    var minr = parseInt(e("shape-width").value);
    var maxr = parseInt(e("shape-height").value);

    // Set window size as image/canvas size
    canvas.width = W;
    canvas.height = H;
    ctx.width = W;
    ctx.height = H;

    // draw second image as background
    ctx.drawImage(img2, 0, 0, canvas.width, canvas.height);

    // save the entire state of the canvas by pushing the current state onto a stack
    ctx.save();

    var r = 20;                   // distance between two circles/ellipses
    var posX = r;                 // init position
    var posY = r;                 // init position

    // marking the circles/ellipses that will be clipped out
    for (var i = 0; i < H / (2 * r); i++) {
        for (var j = 0; j < W / (2 * r); j++) {
            if ((j + i + 1) % 2)                                                  // indexing to make a cool pattern
                if (isCircle)
                    ctx.arc(posX, posY, getRandom(minr, maxr), 0, 2 * Math.PI);   // circle
                else
                    ctx.ellipse(                                                  // ellipse
                        posX, posY,
                        getRandom(minr, maxr),
                        getRandom(minr, maxr),
                        Math.PI / 4,
                        0,
                        2 * Math.PI
                    );

            posX += 2 * r;                             // change X position to right
        }
        posX = r;                                      // change X position at the row beggining
        posY += 2 * r;                                 // change  Y position to the next row
        ctx.moveTo(posX, posY);                        // move (at the beggining of the next row)
    }
    // draw outline stroke if set
    if ($("#outline").is(":checked")) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = parseInt(e("stroke-width").value);
        ctx.stroke();
    }

    // clip the created pattern
    ctx.clip("evenodd");
    // draw image over the pattern
    ctx.drawImage(img1, 0, 0, canvas.width, canvas.height);
    // restore the most recently saved canvas
    ctx.restore();

    // set the result image as background
    e("body").style.backgroundImage =
        "url(" + canvas.toDataURL() + ")";

    // make download button visible => user can download the image now
    $("#download").css("visibility", "visible");
}

// Function randomly creates triangles in the square
function doTriangle(ctx, startx, starty, width, height) {

    // Whole top triangle
    if (Math.random() < 0.75) {
        ctx.moveTo(startx, starty);
        ctx.lineTo(startx, starty + height);
        ctx.lineTo(startx + width, starty + height);
        ctx.lineTo(startx, starty);
    }
    // dividing top triangle
    else {
        if (Math.random() < 0.5) {
            ctx.moveTo(startx, starty);
            ctx.lineTo(startx, starty + height);
            ctx.lineTo(startx + width / 2, starty + height / 2);
            ctx.lineTo(startx, starty);
        } else {
            ctx.moveTo(startx, starty + height);
            ctx.lineTo(startx + width, starty + height);
            ctx.lineTo(startx + width / 2, starty + height / 2);
            ctx.lineTo(startx, starty + height);
        }
    }
    // Whole bottom triangle
    if (Math.random() < 0.01) {
        ctx.moveTo(startx + width, starty);
        ctx.lineTo(startx - 1, starty);
        ctx.lineTo(startx + width, starty + height + 1);
        ctx.lineTo(startx + width, starty);
    }
    // dividing bottom triangle
    else {
        if (Math.random() < 0.5) {
            ctx.moveTo(startx + width, starty);
            ctx.lineTo(startx, starty);
            ctx.lineTo(startx + width / 2, starty + height / 2);
            ctx.lineTo(startx + width, starty);
        } else {
            ctx.moveTo(startx + width, starty);
            ctx.lineTo(startx + width, starty + height);
            ctx.lineTo(startx + width / 2, starty + height / 2);
            ctx.lineTo(startx + width, starty);
        }
    }
}


// Main drawing function for rollage using triangles
function drawTriangles() {
    // Get canvas object and its context
    var canvas = e("rendering");
    var ctx = canvas.getContext("2d");
    // Get color of the stroke
    var strokeColor = e("colorpicker").value;

    // User's window size
    var H = window.screen.height;
    var W = window.screen.width;

    // Set window size as image/canvas size
    canvas.width = W;
    canvas.height = H;
    ctx.width = W;
    ctx.height = H;

    // Get one triangle height and width
    var height = parseInt(e("shape-height").value);
    var width = parseInt(e("shape-width").value);

    // Compute matrix size
    var rows = Math.ceil(H / height);
    var cols = Math.ceil(canvas.width / width);

    // Draw second image to the back
    ctx.drawImage(img2, 0, 0, canvas.width, canvas.height);

    // Save the current context
    ctx.save();
    // Begin new path
    ctx.beginPath();

    // Marking the circles/ellipses that will be clipped out
    for (var col = 0; col <= cols; col++) {
        for (var row = 0; row <= rows; row++) {
            // Compute positions of a square
            var startx = col * width;
            var starty = row * height;
            // Create a triangles in a square
            doTriangle(ctx, startx, starty, width, height);
        }
    }
    // Close the path
    ctx.closePath();

    // Clip the pattern
    ctx.clip("evenodd");

    // Draw the first image to the pattern
    ctx.drawImage(img1, 0, 0, canvas.width, canvas.height);

    // Restore the context
    ctx.restore();

    // Draw outline stroke if set
    if ($("#outline").is(":checked")) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = parseInt(e("stroke-width").value);
        ctx.stroke();
    }

    // Set the result image as background
    e("body").style.backgroundImage =
    "url(" + canvas.toDataURL() + ")";

    // Make download button visible => user can download the image now
    $("#download").css("visibility", "visible");
}
/*********************************************************************************/
/******************* end of Drawing and calcutating functions  *******************/

// Call the initialization function
init();
