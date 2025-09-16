import pygame


pygame.init()

h, w = 8, 11
square_size = 100

traps_cords0 = [ (2,2), (w-3,1), (w-3,2), (w-2,2), (w-4,5)]
energy_cords0 = [ (0,0), (w//2,h//2), (w-1,h-1) ]

traps_cords = [ (2,2), (w-3,1), (w-3,2), (w-2,2), (w-4,5)]
energy_cords = [ (0,0), (w//2,h//2), (w-1,h-1) ]


window_w, window_h = w*square_size, h*square_size
display_surface = pygame.display.set_mode((window_w+400, window_h))
pygame.display.set_caption('bebrik')
clock = pygame.time.Clock()


info_surf = pygame.Surface( (400,window_h) )
info_surf.fill('black')

player_surf = pygame.image.load('images/player2.png')
player_surf = pygame.transform.scale(player_surf, (square_size,square_size) ).convert_alpha()

pivo_surf = pygame.image.load('images/pivo.png')
pivo_surf = pygame.transform.scale(pivo_surf, (square_size,square_size) ).convert_alpha()

energy_surf = pygame.image.load('images/energy.png')
energy_surf = pygame.transform.scale(energy_surf, (square_size,square_size) ).convert_alpha()

trap_surf = pygame.image.load('images/trap.png')
trap_surf = pygame.transform.scale(trap_surf, (square_size,square_size) ).convert_alpha()


class Player:
    def __init__(self):
        self.x = 0
        self.y = h-1
        self.traps_cnt = 0
        self.energy_cnt = 0
        self.moves = 0
        self.won = False

    def move(self, event):
        if event.key == pygame.K_DOWN:
            if self.y < h - 1:
                self.y += 1
                self.moves +=1
        elif event.key == pygame.K_UP:
            if self.y > 0:
                self.y -= 1
                self.moves += 1
        elif event.key == pygame.K_LEFT:
            if self.x > 0:
                self.x -= 1
                self.moves += 1
        elif event.key == pygame.K_RIGHT:
            if self.x < w - 1:
                self.x += 1
                self.moves += 1

    def check_traps(self):
        global traps_cords
        for i in range(len(traps_cords)):
            if (self.x,self.y) == traps_cords[i]:
                self.traps_cnt += 1
                traps_cords.pop(i)
                break
    def check_pivo(self):
        return (self.x == w - 1) and (self.y == 0)

    def check_energy(self):
        global energy_cords
        for i in range(len(energy_cords)):
            if (self.x, self.y) == energy_cords[i]:
                self.energy_cnt += 1
                energy_cords.pop(i)
                break
    def reset(self):
        global traps_cords,energy_cords
        self.x = 0
        self.y = h-1
        self.traps_cnt = 0
        self.energy_cnt = 0
        self.moves = 0
        self.won = False


player = Player()

font = pygame.font.Font('freesansbold.ttf', 32)
episodes_cnt = 1

def run_game():
    global energy_cords,traps_cords,episodes_cnt
    running = True
    while running:
        clock.tick(100)

        if player.won:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        player.reset()
                        traps_cords = traps_cords0.copy()
                        energy_cords = energy_cords0.copy()
                        episodes_cnt += 1

        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT}:
                        player.move(event)

        #drawing the game
        display_surface.fill(color=(255, 255, 0))

        for i in range(h):
            for j in range(w):
                pygame.draw.rect(display_surface,'black',[j*square_size,i*square_size,square_size,square_size], 1)

        display_surface.blit(pivo_surf, ( (w-1)*square_size, 0))

        for x,y in energy_cords:
            display_surface.blit(energy_surf, (x * square_size, y*square_size))

        for x,y in traps_cords:
            display_surface.blit(trap_surf, (x * square_size, y * square_size))

        display_surface.blit(player_surf, (player.x * square_size, player.y * square_size))
        display_surface.blit(info_surf, (w*square_size, 0))


        for i,(item,name) in enumerate( zip([player.traps_cnt, player.energy_cnt, player.moves, episodes_cnt], ['traps','energy','moves', 'episode']) ):
            info = (f'{name}:{ item }')
            text = font.render(info, True, 'white')
            display_surface.blit(text,(w*square_size+20, (i+1)*100))

        if player.check_pivo():
            text1 = font.render('YOU WON!!!', True, 'white')
            text2 = font.render('press Esc to restart', True, 'white')
            display_surface.blit(text1, (w * square_size + 20, 500))
            display_surface.blit(text2, (w * square_size + 20, 600))
            player.won = True

        player.check_traps()
        player.check_energy()

        pygame.display.update()


run_game()
pygame.quit()
