from Tkinter import *
import time
from threading import Thread
import turtle
import math
import random

screen=turtle.Turtle
# turtle for player
player = turtle.Turtle()
player.color("blue")
player.shape("triangle")
player.penup()
player.speed(0)
player.setposition(0,-199)
player.setheading(90) # makes the tip of triangle turn 90 degrees

playermovement= 18 # pixles moves each time




# Choose a number of enemies
numberenemies=5
# creates empty list of enemies
listEnemies=[]
# Add enemies to list
for i in range(numberenemies):
    # creates the invador/enemy
    listEnemies.append(turtle.Turtle())

for enemy in listEnemies:
    enemy.color("Green")
    enemy.shape("circle")
    enemy.penup()
    enemy.speed(0)
    x=random.randint(-200,200) # creates a random spot for turtle
    y=random.randint(100,250)
    enemy.setposition(x,y)

enemymovement=2 # how many pixles enemy moves at a time

# creates a laser for player

laser = turtle.Turtle()
laser.color("red")
laser.shape("triangle")
laser.penup()
laser.speed(0)
laser.setheading(90)
laser.shapesize(0.5,0.5) # makes laser smaller by .5 compared to the player
laser.hideturtle() #hides laser at start of game

lasermovement= 20 # pixles mived by laser

# score count

def kill(score=0):
   if laser.pos()== enemy.pos():
       score+=1
       print("oof")

# timer
timer=4
def countdown():
   global timer # Makes the timer changeable
   timer-=1

laserstate= "ready"
#Move player left and right
def moveleft(*args):
 x = player.xcor() # current x position
 x -=playermovement # subtracts the playermovement
 if x < -330: # if the x value is lower than -357
    x = -330 # x will stop moving
 player.setx(x) # sets the players new location, new x value

def moveright(*args):
 x=player.xcor() # current x position
 x+=playermovement # Adds the playermovement (so it can go the opposite way)
 if x > 320: # If x is greater than 349,
    x = 320 # the x value will stop moving
 player.setx(x) # sets the new x cordinate

def firelaser(*args):
  # states that laserstate is a global if it needs to be changed
  global laserstate # global makes "laserstate" changable
  #move laser just above player
  x=player.xcor() # gets x cordinate of player
  y=player.ycor() +10 # gets y cordinate of player
  laser.setposition(x,y)  # makes laser go 10 pixles above player, so it can be seen
  laser.showturtle()


# checks for a colision
def isCollision(laser,enemy):
    distance= math.sqrt(math.pow(laser.xcor()-enemy.xcor(),2)+math.pow(laser.ycor()-enemy.ycor(),2))
    if distance < 30:
        return True
    else:
        return False
# Checks for a collision between bullet and enemy
if isCollision(laser, enemy):
    # Reset bullet
    laser.hideturtle()
    laserstate="ready"
    laser.setposition(0,-400) # gets bullet out of the way
    # Reset Enemy
    enemy.setposition(-200,250)




# creates key bindings

turtle.listen() # turtle listens to commands below

turtle.onkey(moveleft,"Left") # When the seft arrow key is pushed, goes to movelift def

turtle.onkey(moveright,"Right") # when the right arrow key is pressed, goes to moveright def

turtle.onkey(firelaser,"space")



# MAIN GAME LOOPPPP!!!




while True: #forever, until broken out of,
    for enemy in listEnemies:
        x=enemy.xcor()# gets current x position of enemy
        x += enemymovement # adds the enemymovement, making enemy go right
        # enemy.setx(x) # creates a new x point for ENEMY
        if enemy.xcor()>320: # when the enemy gets to right side of border,
            y = enemy.ycor() # makes enemy go down (y axis)/ gets y cordinate
            y-=20 # how much the enemy will drop
            enemymovement *= -1# makes enemy go back to the left
            enemy.sety(y) # finds new y point
        if enemy.xcor()< -330:
            y= enemy.ycor() # gets y coordinate
            y-=20 # how much enemy will move down when it touches the left side
            enemymovement *= -1 # makes enemy go back to right
            enemy.sety(y) # sets new y point
    # Moves Laser
    y = laser.ycor()
    y += lasermovement
    laser.setY(y)
