
class handler{
    constructor(){
        this.clickTime = 1;
        this.point1 = [0, 0];
        this.point2 = [0, 0];
    }
    handleEvent(event) {
        if (event.target.id !== "vid"){
            throw "wrong click";
        } else {
            switch (this.clickTime){
                case 1:
                    // maybe need to change to pageX[Y]
                    this.point1 = [event.clientX, event.clientY];
                    break;
                case 2:
                    this.point2 = [event.clientX, event.clientY];
                    this.clickTime = 1;
                    this.changeCoords();
                    document.removeEventListener('click');
                    break;
            }

        }
    }

    changeCoords(){
        document.getElementById("line_x1").value = this.point1[0];
        document.getElementById("line_y1").value = this.point1[1];
        document.getElementById("line_x2").value = this.point2[0];
        document.getElementById("line_y2").value = this.point2[1];        
    }

}

function lineClick(){
    let hand = new handler();
    document.addEventListener('click', hand.handleEvent);
}