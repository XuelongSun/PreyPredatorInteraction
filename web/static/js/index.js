const sleep = ms => new Promise(r => setTimeout(r, ms));

const arenaWidthInput = document.getElementById("arenaWidth");
const arenaWidthValue = document.getElementById("arenaWidth-value");
arenaWidthValue.textContent = arenaWidthInput.value;
arenaWidthInput.addEventListener("input", () => {
  arenaWidthValue.textContent = arenaWidthInput.value;
});

const preyNumInput = document.getElementById("preyNum");
const preyNumValue = document.getElementById("preyNum-value");
preyNumValue.textContent = preyNumInput.value;
preyNumInput.addEventListener("input", () => {
  preyNumValue.textContent = preyNumInput.value;
});

const predatorSpeedInput = document.getElementById("predatorSpeed");
const predatorSpeedValue = document.getElementById("predatorSpeed-value");
predatorSpeedValue.textContent = predatorSpeedInput.value;
predatorSpeedInput.addEventListener("input", () => {
  predatorSpeedValue.textContent = predatorSpeedInput.value;
});

const pheromoneWidthInput = document.getElementById("pheromoneWidth");
const pheromoneWidthValue = document.getElementById("pheromoneWidth-value");
pheromoneWidthValue.textContent = pheromoneWidthInput.value;
pheromoneWidthInput.addEventListener("input", () => {
  pheromoneWidthValue.textContent = pheromoneWidthInput.value;
});

function logMessage(message) {
  var date = new Date();
  var datetime = date.toLocaleString();
  var logger = document.getElementById("status");
  var p = document.createElement("p");
  p.innerHTML = datetime + " - " + message;
  logger.appendChild(p);
}


const plot_layout = {
  font:{
    color:"white",
    size:10
  },
  margin: {
    l: 30,
    r: 30,
    b: 30,
    t: 30,
    pad: 10
  },
  paper_bgcolor: 'rgba(49, 48, 48, 0.5)',
  plot_bgcolor: 'rgba(49, 48, 48, 0.5)',
};

var plotl1 = document.getElementById("plot-line-1");
var plotl2 = document.getElementById("plot-line-2");
var plotb1 = document.getElementById("plot-bar-1");
var plotb2 = document.getElementById("plot-bar-2");

function rebuild_plots(){
  var plotData = [
    {x: [], y: [], type: "line", line: {color: 'cyan', width: 2, shape: 'spline'}, name:'Prey'},
    {x: [], y: [], type: "line", line: {color: 'blue', width: 2, shape: 'spline'}, name:'Predator'}
  ];
  
  Plotly.newPlot(plotl1, plotData, {...plot_layout, title:"Energy"});

  var plotData = [{x: [], y: [], type: "line"}];
  Plotly.newPlot(plotl2, plotData, {...plot_layout, title:"Data"});

  var plotData = [{x: [], y: [], type: "bar"}];
  Plotly.react(plotb1, plotData, {...plot_layout, title:"Prey-f_gather Distribution"});

  var plotData = [{x: [], y: [], type: "bar"}];
  Plotly.react(plotb2, plotData, {...plot_layout, title:"Prey-f_avoid Distribution"});
}

rebuild_plots()

const canvas = document.getElementById('arena-image');
const ctx = canvas.getContext('2d');


// var socket = io.connect();
var socket = io.connect("http://192.168.3.99:8080");
// const socket = io.connect("http://192.168.31.141:8080");
// const socket = io.connect('http://localhost:8080');

socket.on("update_plot", function (results) {
  Plotly.extendTraces(plotl1, {x:[[results.x], [results.x]], y:[[results.y11],[results.y12]]}, [0, 1]);

  // Plotly.update(plotb1, {x:results.x_gather, y:results.y_gather}, {}, [0]);
  // Plotly.update(plotb2, {x:results.x_avoid, y:results.y_avoid}, {}, [0]);
  Plotly.react(
    plotb1,
    [{x:results.x_gather, y:results.y_gather, type:"bar"}],
    {...plot_layout, title:"Prey-f_gather Distribution"}
  );

  Plotly.react(
    plotb2,
    [{x:results.x_avoid, y:results.y_avoid, type:"bar"}],
    {...plot_layout, title:"Prey-f_avoid Distribution"}
  );
  // console.log({x:results.x_gather, y:results.y_gather})
});

socket.on("simulation_started", function () {
  // $("#status").text("Simulation running...");
  // $("#status").textContent += "Simulation running...\n";
  logMessage("Simulation running...");
  $("#startButton").prop("disabled", true);
  $("#stopButton").prop("disabled", false);
  rebuild_plots()
});

socket.on("simulation_stopped", function () {
  // $("#status").text("Simulation stopped.");
  // $("#status").textContent += "Simulation stopped.\n";
  logMessage("Simulation stopped.");
  $("#startButton").prop("disabled", false);
  $("#stopButton").prop("disabled", true);
});

// Receive the base64-encoded image from the Flask SocketIO server and display it on the canvas
socket.on('update_image', function(data) {
  // arena image
  const img = new Image();
  img.onload = function() {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = 'data:image/jpeg;base64,' + data.image;
});

function startSimulation() {
  // $("#status").text("Starting simulation...");
  // $("#status").textContent += "Starting simulation...\n";
  
  parameters = {
    'arena_width':arenaWidthInput.value,
    'prey_num':preyNumInput.value,
    'predator_speed':predatorSpeedInput.value,
    'pheromone_width':pheromoneWidthInput.value}
  logMessage("Starting simulation with parameters: " + JSON.stringify(parameters))
  socket.emit("start_simulation", parameters);
}

function stopSimulation() {
  // $("#status").text("Stopping simulation...");
  // $("#status").textContent += "Stopping simulation...\n";
  logMessage("Stopping simulation...")
  socket.emit("stop_simulation");
}