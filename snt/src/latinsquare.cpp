#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/set.hh>
#include <gecode/gist.hh>
#include <fstream>


using namespace Gecode;

class GolfOptions : public Options {
protected:
    Driver::IntOption _w; // Number of weeks
    Driver::IntOption _g; // Number of groups
    Driver::IntOption _s; // Number of players per group
public:
    // Constructor
    GolfOptions(void) : Options("Latin Square"), _w("w","number of weeks",9), _g("g","number of groups",8), _s("s","number of players per group",4) {
        add(_w);
        add(_g);
        add(_s);
    }
    // Return number of weeks
    int w(void) const { return _w.value(); }
    // Return number of groups
    int g(void) const { return _g.value(); }
    // Return number of players per group
    int s(void) const { return _s.value(); }
};


class LatinSquare : public Script {
    public:
    int g; // Number of groups in a week
    int s; // Number of players in a group
    int w; // Number of weeks

    // The sets representing the groups
    IntVarArray groups;

    // Actual model
    LatinSquare(const GolfOptions& opt) : Script(opt), g(opt.g()), s(opt.s()), w(opt.w()) {
        groups = IntVarArray(*this, g*g, 1, g);
        Matrix<IntVarArray> GS1(groups,g,g);

        // fix the first row
        int beginValues[] = {2,s,1,3,4};
        for (int i = 0; i < g; i++) {
            GS1(i,0) = IntVar(*this, (i+1), (i+1));
            GS1(i,1) = (i < 5) ? IntVar(*this, beginValues[i], beginValues[i]) : IntVar(*this, i, i);
        }

        // fix last rows
        int endValues[] = {2,1,3,4,5};
        for (int row = g-1, i=2; row > 1; row--,i++) {
            for (int col = i,j=0; col < g; col++,j++) {
                GS1(col, row) = (j < 5) ? IntVar(*this, endValues[j], endValues[j]) : IntVar(*this, j+1, j+1);
            }
        }

        // the entries of GS1 are symmetric with respect to the main diagonal. 
        for (int row = 0; row < g; row++) 
            for (int col = 0; col < g; col++) 
                rel(*this, GS1(col,row), IRT_EQ, GS1(row, col));
            
        // the main diagonal of the submatrix GS1 is pairwise distinct
        IntVarArgs diagonal;
        for (int i = 0; i < g; i++) 
            diagonal << GS1(i,i); 
        distinct(*this, diagonal);
        
        // each col and row is pairwise distinct
        for (int i = 0; i < g; i++) {
            distinct(*this, GS1.row(i));
            distinct(*this, GS1.col(i));
        }
       
        branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
    }

    // Print solution
    virtual void print(std::ostream& os) const {
        std::ofstream output ("square.txt");
        Matrix<IntVarArray> schedule(groups,g,g);
        for (int j = 0; j < g; j++) {
            os << schedule.row(j) << std::endl;
        }
        output << "{\n";
        for (int j = 0; j < g; j++) {
            for (int i = 0; i < g; i++) {
                output << schedule(i,j) << ", ";
            }
            output << "\n";
        }
        output << "}";

    }

    // Constructor for copying \a s
    LatinSquare(LatinSquare& s) : Script(s), g(s.g), s(s.s), w(s.w) {
        groups.update(*this, s.groups);
    }

    // Copy during cloning
    virtual Space* copy(void) {
        return new LatinSquare(*this);
    }
 
};

int
main(int argc, char* argv[]) {
    GolfOptions opt;
    opt.ipl(IPL_BND);
    opt.solutions(1);
    opt.parse(argc,argv);

    Script::run<LatinSquare,DFS,GolfOptions>(opt);
    return 0;
}
